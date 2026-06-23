"""SQLite-backed job/download state store.

This replaces the old in-memory ``_jobs`` / ``_downloads`` dicts. State lives in
a single SQLite file inside the bind-mounted ``data/`` dir (``data/voxset.db``),
so it survives a container/process restart — no extra service, no image bloat
(``sqlite3`` is in the Python standard library).

Record shapes are kept byte-for-byte compatible with the old dicts so the
frontend and the FastAPI routes need no changes:

  job      -> {status, stage, progress, logs[], srt, error, file, title, cancel}
  download -> {status, items:[{id,title,status,progress,file,error}], error}

The whole record is stored as a JSON blob keyed by id, plus a ``status`` column
for cheap "list the active ones" queries. All writes go through a single lock so
read-modify-write updates (progress, log append, cancel flag) stay atomic across
the FastAPI request threads and the worker threads.
"""
import json
import os
import sqlite3
import threading

from src import config

_JOB_TERMINAL = {"done", "error", "canceled", "interrupted"}

_conn = None
_lock = threading.Lock()


def _db():
    global _conn
    if _conn is None:
        os.makedirs(config.DATA_DIR, exist_ok=True)
        path = os.path.join(config.DATA_DIR, "voxset.db")
        _conn = sqlite3.connect(path, check_same_thread=False)
        _conn.execute("PRAGMA journal_mode=WAL")
        _conn.execute("CREATE TABLE IF NOT EXISTS jobs "
                      "(id TEXT PRIMARY KEY, status TEXT, rec TEXT)")
        _conn.execute("CREATE TABLE IF NOT EXISTS downloads "
                      "(id TEXT PRIMARY KEY, status TEXT, rec TEXT)")
        _conn.commit()
    return _conn


def _put(table, rec_id, rec):
    with _lock:
        db = _db()
        db.execute(
            f"INSERT INTO {table}(id, status, rec) VALUES(?,?,?) "
            f"ON CONFLICT(id) DO UPDATE SET status=excluded.status, rec=excluded.rec",
            (rec_id, rec.get("status"), json.dumps(rec)))
        db.commit()


def _get(table, rec_id):
    with _lock:
        row = _db().execute(f"SELECT rec FROM {table} WHERE id=?", (rec_id,)).fetchone()
    return json.loads(row[0]) if row else None


def _mutate(table, rec_id, fn):
    """Atomically read-modify-write a record (single lock held the whole time)."""
    with _lock:
        db = _db()
        row = db.execute(f"SELECT rec FROM {table} WHERE id=?", (rec_id,)).fetchone()
        if row is None:
            return None
        rec = json.loads(row[0])
        fn(rec)
        db.execute(f"UPDATE {table} SET status=?, rec=? WHERE id=?",
                   (rec.get("status"), json.dumps(rec), rec_id))
        db.commit()
        return rec


# --------------------------------------------------------------------------- #
# Transcription jobs
# --------------------------------------------------------------------------- #
def create_job(job_id, file=None, title=None):
    rec = {"status": "queued", "stage": "queued", "progress": 0.0,
           "logs": [], "srt": None, "error": None, "file": file, "title": title,
           "cancel": False}
    _put("jobs", job_id, rec)
    return rec


def get_job(job_id):
    return _get("jobs", job_id)


def update_job(job_id, **fields):
    return _mutate("jobs", job_id, lambda r: r.update(fields))


def append_log(job_id, message, stage=None, progress=None):
    def fn(r):
        if stage is not None:
            r["stage"] = stage
        if progress is not None:
            r["progress"] = progress
        r.setdefault("logs", []).append(message)
    return _mutate("jobs", job_id, fn)


def list_active_jobs():
    """Queued/running jobs for the UI to re-attach after a reload."""
    with _lock:
        rows = _db().execute(
            "SELECT id, rec FROM jobs WHERE status IN ('queued','running')").fetchall()
    out = []
    for jid, rec in rows:
        r = json.loads(rec)
        out.append({"id": jid, "file": r.get("file"),
                    "title": r.get("title") or r.get("file"),
                    "status": r["status"], "stage": r.get("stage"),
                    "progress": r.get("progress", 0.0)})
    return out


# --- cooperative cancellation ---------------------------------------------- #
def request_cancel(job_id):
    _mutate("jobs", job_id, lambda r: r.__setitem__("cancel", True))


def is_canceled(job_id):
    rec = get_job(job_id)
    return bool(rec and rec.get("cancel"))


def clear_cancel(job_id):
    _mutate("jobs", job_id, lambda r: r.__setitem__("cancel", False))


# --- restart reconciliation ------------------------------------------------- #
def reconcile():
    """Mark jobs/downloads orphaned by a restart.

    Workers run as in-process threads, so a fresh process has none running: any
    job still recorded as queued/running, or any download not yet done, died
    with the previous process and is flagged so the UI shows it (instead of it
    hanging forever) and the user can retry.
    """
    with _lock:
        jids = [r[0] for r in _db().execute(
            "SELECT id FROM jobs WHERE status IN ('queued','running')").fetchall()]
        dids = [r[0] for r in _db().execute(
            "SELECT id FROM downloads WHERE status!='done'").fetchall()]

    def _job_interrupted(r):
        r["status"] = "interrupted"
        r["stage"] = "interrupted"
        r["error"] = "Server restarted — job was interrupted. Please retry."

    def _dl_interrupted(r):
        for it in r.get("items", []):
            if it.get("status") not in ("done", "error"):
                it["status"] = "error"
                it["error"] = "Server restarted — download interrupted."
        r["status"] = "done"

    for jid in jids:
        _mutate("jobs", jid, _job_interrupted)
    for did in dids:
        _mutate("downloads", did, _dl_interrupted)
    return jids, dids


# --------------------------------------------------------------------------- #
# Download jobs
# --------------------------------------------------------------------------- #
def create_download(dl_id, items):
    rec = {"status": "queued",
           "items": [{"id": it["id"], "title": it["title"], "status": "pending",
                      "progress": 0.0, "file": None, "error": None} for it in items],
           "error": None}
    _put("downloads", dl_id, rec)
    return rec


def get_download(dl_id):
    return _get("downloads", dl_id)


def update_download(dl_id, **fields):
    return _mutate("downloads", dl_id, lambda r: r.update(fields))


def update_download_item(dl_id, idx, **fields):
    return _mutate("downloads", dl_id, lambda r: r["items"][idx].update(fields))


def list_active_downloads():
    """Not-done downloads for the UI to re-attach after a reload."""
    with _lock:
        rows = _db().execute(
            "SELECT id, status FROM downloads WHERE status!='done'").fetchall()
    return [{"id": did, "status": status} for did, status in rows]
