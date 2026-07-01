"""Job runner: spawns a subprocess per job and supervises it.

Submission creates a registry entry (web/jobs.py) and starts a daemon
*supervisor* thread. The supervisor waits on a per-kind semaphore (so at most
MAX_*_JOBS of each kind run at once; a GPU pipeline and a download overlap),
spawns the worker subprocess, drains its event queue into the registry (which
fans each new snapshot out to SSE clients), and finalizes the job's state when
the process exits.

Cancellation:
  * transcribe — terminate the subprocess (frees the GPU immediately).
  * download   — cooperative: push the item index (or "all") onto the worker's
                 control queue; the worker aborts at its next progress tick.
"""
import multiprocessing as mp
import os
import queue as _q
import threading

from src import config
from web import jobs
from web.workers import run_transcribe, run_download, run_translate

# spawn (not fork): the parent never initializes CUDA, and spawn behaves the
# same on Windows and Linux — the child re-imports its target module fresh.
_ctx = mp.get_context("spawn")

_gpu_sem = threading.Semaphore(max(1, config.MAX_TRANSCRIBE_JOBS))
_dl_sem = threading.Semaphore(max(1, config.MAX_DOWNLOAD_JOBS))

# Coarse progress fractions for the (callback-less) whisperx stages.
_STAGE_FRAC = {"info": 0.40, "load": 0.45, "transcribe": 0.60,
               "align": 0.75, "diarize": 0.88, "write": 0.96, "done": 1.0}

# Supervisor-side handles (never put in the public snapshot).
_ctl_lock = threading.Lock()
_procs = {}        # jid -> subprocess
_controls = {}     # jid -> control Queue (downloads)
_cancel = set()    # jids with a whole-job cancel requested


# --------------------------------------------------------------------------- #
# small handle helpers
# --------------------------------------------------------------------------- #
def _set(d, jid, v):
    with _ctl_lock:
        d[jid] = v


def _pop(d, jid):
    with _ctl_lock:
        return d.pop(jid, None)


def _get(d, jid):
    with _ctl_lock:
        return d.get(jid)


def _request_cancel(jid):
    with _ctl_lock:
        _cancel.add(jid)


def _clear_cancel(jid):
    with _ctl_lock:
        _cancel.discard(jid)


def _is_cancel_requested(jid):
    with _ctl_lock:
        return jid in _cancel


# --------------------------------------------------------------------------- #
# Transcription
# --------------------------------------------------------------------------- #
def submit_transcribe(file, title, params):
    rec = {"status": "queued", "stage": "queued", "progress": 0.0, "logs": [],
           "srt": None, "error": None, "file": file, "title": title, "params": params}
    jid = jobs.create("transcribe", rec)
    threading.Thread(target=_supervise_transcribe, args=(jid, params), daemon=True).start()
    return jid


def retry_transcribe(jid):
    _clear_cancel(jid)
    snap = jobs.update(jid, status="queued", stage="queued", progress=0.0,
                       srt=None, error=None, logs=[])
    params = snap.get("params")
    threading.Thread(target=_supervise_transcribe, args=(jid, params), daemon=True).start()


def cancel_transcribe(jid):
    snap = jobs.get(jid)
    if snap is None:
        return None
    if snap["status"] in jobs.TERMINAL:
        return snap["status"]
    _request_cancel(jid)
    proc = _get(_procs, jid)
    if proc is not None and proc.is_alive():
        proc.terminate()                 # supervisor finalizes to "canceled"
    elif snap["status"] == "queued":
        jobs.update(jid, status="canceled", stage="canceled")
    cur = jobs.get(jid)
    return cur["status"] if cur else "canceled"


def _supervise_transcribe(jid, params):
    _gpu_sem.acquire()
    try:
        if _is_cancel_requested(jid):
            jobs.update(jid, status="canceled", stage="canceled")
            return
        events = _ctx.Queue()
        proc = _ctx.Process(target=run_transcribe, args=(events, params), daemon=True)
        _set(_procs, jid, proc)
        proc.start()
        jobs.update(jid, status="running", stage="starting")
        terminal = _drain(jid, "transcribe", proc, events)
        _join(proc)
        _finalize(jid, "transcribe", terminal)
    finally:
        _gpu_sem.release()
        _pop(_procs, jid)


# --------------------------------------------------------------------------- #
# Translation (shares the GPU semaphore with transcription)
# --------------------------------------------------------------------------- #
def submit_translate(srt_name, base, src, target, title=None):
    params = {"srt_name": srt_name, "base": base, "src": src, "target": target}
    rec = {"status": "queued", "stage": "queued", "progress": 0.0, "logs": [],
           "srt": None, "error": None, "title": title or base, "target": target,
           "params": params}
    jid = jobs.create("translate", rec)
    threading.Thread(target=_supervise_translate, args=(jid, params), daemon=True).start()
    return jid


def cancel_translate(jid):
    return cancel_transcribe(jid)        # same mechanism: terminate the process


def _supervise_translate(jid, params):
    _gpu_sem.acquire()
    try:
        if _is_cancel_requested(jid):
            jobs.update(jid, status="canceled", stage="canceled")
            return
        events = _ctx.Queue()
        proc = _ctx.Process(target=run_translate, args=(events, params), daemon=True)
        _set(_procs, jid, proc)
        proc.start()
        jobs.update(jid, status="running", stage="translate")
        terminal = _drain(jid, "translate", proc, events)
        _join(proc)
        _finalize(jid, "translate", terminal)
    finally:
        _gpu_sem.release()
        _pop(_procs, jid)


# --------------------------------------------------------------------------- #
# Downloads
# --------------------------------------------------------------------------- #
def submit_download(items, mode, quality):
    rec = {"status": "queued", "mode": mode, "quality": quality, "error": None,
           "items": [{"id": it["id"], "title": it["title"], "url": it.get("url"),
                      "playlist": it.get("playlist"), "status": "pending",
                      "progress": 0.0, "file": None, "error": None, "cancel": False}
                     for it in items]}
    jid = jobs.create("download", rec)
    _start_download(jid, list(range(len(items))))
    return jid


def retry_download_item(jid, idx):
    _clear_cancel(jid)

    def fn(r):
        r["items"][idx].update(status="pending", progress=0.0, error=None,
                               file=None, cancel=False)
    jobs.mutate(jid, fn)
    jobs.update(jid, status="running")
    _start_download(jid, [idx])


def cancel_download(jid):
    snap = jobs.get(jid)
    if snap is None:
        return None
    if snap["status"] in jobs.TERMINAL:
        return snap["status"]
    _request_cancel(jid)

    def fn(r):                              # flip still-pending items at once
        for it in r.get("items", []):
            if it.get("status") == "pending":
                it["status"] = "canceled"
    jobs.mutate(jid, fn)
    control = _get(_controls, jid)
    if control is not None:
        control.put("all")
    if snap["status"] == "queued":
        jobs.update(jid, status="canceled")
    return "canceling"


def cancel_download_item(jid, idx):
    def fn(r):
        it = r["items"][idx]
        it["cancel"] = True
        if it.get("status") == "pending":
            it["status"] = "canceled"
    jobs.mutate(jid, fn)
    control = _get(_controls, jid)
    if control is not None:
        control.put(idx)


def _start_download(jid, indices):
    control = _ctx.Queue()
    _set(_controls, jid, control)
    threading.Thread(target=_supervise_download, args=(jid, indices, control),
                     daemon=True).start()


def _supervise_download(jid, indices, control):
    _dl_sem.acquire()
    try:
        snap = jobs.get(jid)
        if snap is None:
            return
        if _is_cancel_requested(jid):
            jobs.update(jid, status="canceled")
            return
        items = [(idx, snap["items"][idx]) for idx in indices
                 if 0 <= idx < len(snap["items"])]
        events = _ctx.Queue()
        proc = _ctx.Process(target=run_download,
                            args=(events, control, items, snap["mode"], snap["quality"]),
                            daemon=True)
        _set(_procs, jid, proc)
        proc.start()
        jobs.update(jid, status="running")
        terminal = _drain(jid, "download", proc, events)
        _join(proc)
        _finalize(jid, "download", terminal)
    finally:
        _dl_sem.release()
        _pop(_procs, jid)
        _pop(_controls, jid)


# --------------------------------------------------------------------------- #
# Shared supervision helpers
# --------------------------------------------------------------------------- #
def _drain(jid, kind, proc, events):
    """Pump worker events into the registry until the worker signals a terminal
    event (done/error/finished) or the process dies. Returns the terminal event
    (or None if the process died without one)."""
    while True:
        try:
            ev = events.get(timeout=0.3)
        except _q.Empty:
            if not proc.is_alive():
                return None
            continue
        if ev.get("t") in ("done", "error", "finished"):
            return ev
        _apply_event(jid, kind, ev)


def _apply_event(jid, kind, ev):
    if kind in ("transcribe", "translate"):
        if ev.get("t") == "log":
            stage = ev.get("stage")
            fields = {}
            if stage:
                fields["stage"] = stage
            # transcribe maps stage->fraction; translate carries an explicit one
            frac = ev.get("progress")
            if frac is None and kind == "transcribe":
                frac = _STAGE_FRAC.get(stage)
            if frac is not None:
                fields["progress"] = frac
            jobs.update(jid, log=ev.get("msg"), **fields)
    else:  # download
        if ev.get("t") == "item":
            idx = ev["idx"]
            patch = {k: v for k, v in ev.items() if k not in ("t", "idx")}

            def fn(r, idx=idx, patch=patch):
                if 0 <= idx < len(r.get("items", [])):
                    r["items"][idx].update(patch)
            jobs.mutate(jid, fn)


def _finalize(jid, kind, terminal):
    if jobs.get(jid) is None:
        return
    if kind in ("transcribe", "translate"):
        if terminal and terminal.get("t") == "done":
            jobs.update(jid, status="done", stage="done", progress=1.0,
                        srt=terminal.get("srt"))
        elif terminal and terminal.get("t") == "error":
            err = terminal.get("error") or "Job failed"
            jobs.update(jid, status="error", stage="error",
                        log=f"ERROR: {err}", error=err)
        elif _is_cancel_requested(jid):
            jobs.update(jid, status="canceled", stage="canceled",
                        log="Job canceled by user.")
        else:
            jobs.update(jid, status="error", stage="error",
                        error="Worker process exited unexpectedly.")
    else:  # download
        cancel_all = bool(terminal and terminal.get("cancel_all")) or _is_cancel_requested(jid)
        if terminal is None:                # process died without finishing
            def fn(r):
                for it in r.get("items", []):
                    if it.get("status") not in ("done", "error", "canceled"):
                        if cancel_all:
                            it["status"] = "canceled"
                        else:
                            it["status"] = "error"
                            it["error"] = "Worker process exited unexpectedly."
            jobs.mutate(jid, fn)
        jobs.update(jid, status="canceled" if cancel_all else "done")


def _join(proc, timeout=5):
    proc.join(timeout)
    if proc.is_alive():
        proc.kill()
        proc.join(2)
