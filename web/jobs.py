"""In-memory job registry with a pub/sub event stream.

This replaces the old SQLite store. A long-lived server process keeps every
job's state in memory; clients tail it live over SSE (see web/app.py). The
pieces of the pattern:

  * Job registry        — ``_jobs`` maps id -> _Job (the snapshot + subscribers).
  * Finite state machine — status moves queued -> running -> done|error|canceled.
  * Ring buffer          — each job's ``logs`` is bounded (oldest lines drop).
  * Publish/subscribe    — ``update`` fans the new snapshot out to every live
                           subscriber queue (one per SSE client).
  * Snapshot + live tail — ``subscribe`` atomically returns the current snapshot
                           AND a queue fed by all *future* updates, so a client
                           that attaches late sees the current state and then
                           every subsequent change, with no gap and no dupes.

State is intentionally NOT persisted: a process restart starts empty (the media
library and datasets on disk are unaffected). Terminal jobs are pruned past a
cap so memory doesn't grow without bound.
"""
import copy
import queue
import threading
import uuid

from src import config

TERMINAL = {"done", "error", "canceled", "interrupted"}
ACTIVE = {"queued", "running"}

_LOG_CAP = 500            # ring buffer: keep only the last N log lines per job

_lock = threading.RLock()
_jobs = {}                # id -> _Job  (insertion-ordered; Py3.7+ dict)


class _Job:
    __slots__ = ("id", "kind", "rec", "subscribers")

    def __init__(self, jid, kind, rec):
        self.id = jid
        self.kind = kind
        self.rec = rec               # public snapshot dict
        self.subscribers = []        # list[queue.Queue] — one per SSE client


def create(kind, rec):
    """Register a new job. `rec` is the initial public snapshot (sans id/kind)."""
    jid = uuid.uuid4().hex[:12]
    rec = dict(rec)
    rec.setdefault("status", "queued")
    rec["id"] = jid
    rec["kind"] = kind
    with _lock:
        _jobs[jid] = _Job(jid, kind, rec)
        _fanout(jid, rec)
        _prune()
    return jid


def get(jid):
    """Return a deep copy of the job snapshot, or None."""
    with _lock:
        job = _jobs.get(jid)
        return copy.deepcopy(job.rec) if job else None


def exists(jid):
    with _lock:
        return jid in _jobs


def update(jid, log=None, **fields):
    """Atomically mutate a job's snapshot and broadcast it to subscribers.

    `log` (optional) is appended to the bounded log ring buffer. `fields` are
    merged into the snapshot. Returns the new snapshot, or None if unknown.
    """
    with _lock:
        job = _jobs.get(jid)
        if job is None:
            return None
        rec = job.rec
        if log is not None:
            logs = rec.setdefault("logs", [])
            logs.append(log)
            if len(logs) > _LOG_CAP:
                del logs[:len(logs) - _LOG_CAP]
        rec.update(fields)
        snap = copy.deepcopy(rec)
        _fanout(jid, snap)
        return snap


def mutate(jid, fn):
    """Read-modify-write the snapshot under the lock, then broadcast it."""
    with _lock:
        job = _jobs.get(jid)
        if job is None:
            return None
        fn(job.rec)
        snap = copy.deepcopy(job.rec)
        _fanout(jid, snap)
        return snap


def list_active(kind=None):
    """Snapshots of queued/running jobs (optionally of one kind), oldest first."""
    with _lock:
        return [copy.deepcopy(j.rec) for j in _jobs.values()
                if j.rec.get("status") in ACTIVE and (kind is None or j.kind == kind)]


# --------------------------------------------------------------------------- #
# Pub/sub: snapshot + live tail
# --------------------------------------------------------------------------- #
def subscribe(jid):
    """Atomically return (snapshot, queue). The queue receives every snapshot
    produced *after* this call — so the caller can send the snapshot first and
    then drain the queue without missing or duplicating an update.

    Returns (None, None) if the job is unknown.
    """
    with _lock:
        job = _jobs.get(jid)
        if job is None:
            return None, None
        q = queue.Queue()
        job.subscribers.append(q)
        return copy.deepcopy(job.rec), q


def unsubscribe(jid, q):
    with _lock:
        job = _jobs.get(jid)
        if job and q in job.subscribers:
            job.subscribers.remove(q)


def _fanout(jid, snap):
    """Push a snapshot to every subscriber of `jid`. Caller holds _lock."""
    job = _jobs.get(jid)
    if not job:
        return
    for q in job.subscribers:
        q.put(snap)


def _prune():
    """Evict oldest terminal jobs once the registry exceeds JOB_HISTORY.

    Caller holds _lock. Active jobs are never evicted; a job with live
    subscribers is kept so its SSE clients still drain cleanly.
    """
    cap = max(1, config.JOB_HISTORY)
    if len(_jobs) <= cap:
        return
    for jid in list(_jobs):
        if len(_jobs) <= cap:
            break
        job = _jobs[jid]
        if job.rec.get("status") in TERMINAL and not job.subscribers:
            del _jobs[jid]
