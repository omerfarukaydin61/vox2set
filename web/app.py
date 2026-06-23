"""FastAPI web interface for VoxSet — a speech dataset builder.

Two decoupled stages, each run in background threads inside this single process:
  * Downloads     — fetch a single video or a whole playlist, as audio or video,
                    into the data/ library (one download job at a time).
  * Transcription — pick a file from the library and run the whisperx pipeline
                    (transcribe/align/diarize) into a speaker-labeled speech
                    dataset, stored as an .srt (one GPU job at a time).

Job state is persisted to SQLite (web/store.py → data/voxset.db), so it survives
a container/process restart: on startup any job/download left mid-flight (its
thread died with the old process) is reconciled to `interrupted`/failed so the
UI shows it instead of hanging, and the user retries. The frontend polls
/api/downloads/{id} and /api/jobs/{id} for live progress, /api/library for the
file list, and /api/gpu for the GPU monitor.
"""
import csv
import io
import os
import shutil
import subprocess
import tempfile
import threading
import uuid

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTask

from src import config
from src.downloader import fetch_info, download_media, thumb_for
from src.pipeline import run_pipeline
from src.util import parse_srt
from web import store
from web.meta import meta_path, read_meta, write_meta

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(HERE, "static")

app = FastAPI(title="VoxSet")

AUDIO_EXTS = {".mp3", ".m4a", ".wav", ".opus", ".flac", ".aac", ".ogg"}
VIDEO_EXTS = {".mp4", ".mkv", ".webm", ".mov", ".avi"}

_run_lock = threading.Lock()         # one GPU pipeline at a time
_dl_lock = threading.Lock()          # one download job at a time

# Coarse progress fractions for the (callback-less) whisperx stages.
_STAGE_FRAC = {"info": 0.40, "load": 0.45, "transcribe": 0.60,
               "align": 0.75, "diarize": 0.88, "write": 0.96, "done": 1.0}


def _migrate_flat_layout():
    """Move any pre-existing flat data/ files into media/ and subtitles/.

    Older versions dumped media, .meta.json sidecars and subtitle-*.srt all in
    the data/ root. This relocates them once so an upgrade doesn't "lose" the
    library. voxset.db, .gitkeep and stray *.part files stay at the root.
    """
    d = config.DATA_DIR
    if not os.path.isdir(d):
        return
    os.makedirs(config.MEDIA_DIR, exist_ok=True)
    os.makedirs(config.SUBS_DIR, exist_ok=True)
    for f in os.listdir(d):
        src = os.path.join(d, f)
        if not os.path.isfile(src):
            continue
        ext = os.path.splitext(f)[1].lower()
        if f.startswith("subtitle-") and f.endswith(".srt"):
            dst = os.path.join(config.SUBS_DIR, f)
        elif ext in AUDIO_EXTS or ext in VIDEO_EXTS or f.endswith(".meta.json"):
            dst = os.path.join(config.MEDIA_DIR, f)
        else:
            continue
        if not os.path.exists(dst):
            try:
                shutil.move(src, dst)
            except OSError:
                pass


@app.on_event("startup")
def _recover():
    """Tidy the data layout, then flag jobs orphaned by a previous process."""
    try:
        _migrate_flat_layout()
    except Exception:
        pass
    try:
        store.reconcile()
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Library helpers
# --------------------------------------------------------------------------- #
def _list_library():
    if not os.path.isdir(config.MEDIA_DIR):
        return []
    srts = ([s for s in os.listdir(config.SUBS_DIR) if s.lower().endswith(".srt")]
            if os.path.isdir(config.SUBS_DIR) else [])
    items = []
    for f in os.listdir(config.MEDIA_DIR):
        ext = os.path.splitext(f)[1].lower()
        if ext not in AUDIO_EXTS and ext not in VIDEO_EXTS:
            continue
        base = os.path.splitext(f)[0]
        meta = read_meta(base)
        my_srts = sorted(s for s in srts
                         if s.startswith("subtitle-") and s.endswith(f"-{base}.srt"))
        items.append({
            "name": f,
            "title": meta.get("title") or base,
            "kind": "audio" if ext in AUDIO_EXTS else "video",
            "ext": ext.lstrip("."),
            "size": os.path.getsize(os.path.join(config.MEDIA_DIR, f)),
            "duration": meta.get("duration"),
            "thumbnail": meta.get("thumbnail"),
            "playlist": meta.get("playlist"),
            "srts": my_srts,
        })
    items.sort(key=lambda x: x["title"].lower())
    return items


# --------------------------------------------------------------------------- #
# Download worker (one at a time)
# --------------------------------------------------------------------------- #
def _download_worker(dl_id, items, mode, quality):
    with _dl_lock:
        store.update_download(dl_id, status="running")
        for idx, it in enumerate(items):
            try:
                store.update_download_item(dl_id, idx, status="downloading", progress=0.0)

                last = {"pct": -1}

                def hook(d, idx=idx, last=last):
                    if d.get("status") == "downloading":
                        total = d.get("total_bytes") or d.get("total_bytes_estimate")
                        if total:
                            frac = d.get("downloaded_bytes", 0) / total
                            pct = int(frac * 100)
                            if pct != last["pct"]:        # throttle DB writes
                                last["pct"] = pct
                                store.update_download_item(dl_id, idx, progress=frac)
                    elif d.get("status") == "finished":
                        store.update_download_item(dl_id, idx, progress=1.0,
                                                   status="processing")

                path, info = download_media(
                    it["url"], config.MEDIA_DIR, mode=mode, quality=quality,
                    codec=config.AUDIO_CODEC, progress_hook=hook)

                base = os.path.splitext(os.path.basename(path))[0]
                meta = {
                    "title": info.get("title") or it["title"],
                    "duration": info.get("duration"),
                    "kind": mode,
                    "url": it["url"],
                    "thumbnail": thumb_for(info),
                }
                if it.get("playlist"):
                    meta["playlist"] = it["playlist"]
                write_meta(base, meta)
                store.update_download_item(dl_id, idx, status="done", progress=1.0,
                                           file=os.path.basename(path))
            except Exception as exc:  # one bad item shouldn't kill the rest
                store.update_download_item(dl_id, idx, status="error", error=str(exc))
        store.update_download(dl_id, status="done")


# --------------------------------------------------------------------------- #
# Transcription worker (one at a time)
# --------------------------------------------------------------------------- #
class _JobCancelled(Exception):
    """Raised inside the pipeline's progress callback to abort a job."""


def _worker(job_id, params):
    with _run_lock:  # serialize GPU work
        # the job may have been canceled while queued (waiting for the lock)
        if store.is_canceled(job_id):
            store.update_job(job_id, status="canceled", stage="canceled")
            return
        store.update_job(job_id, status="running", stage="starting")

        def prog(stage, msg):
            # cooperative cancellation: checked at every pipeline stage boundary
            if store.is_canceled(job_id):
                raise _JobCancelled()
            store.append_log(job_id, msg, stage=stage, progress=_STAGE_FRAC.get(stage))

        # HF_TOKEN and the media/subtitle dirs come from the server env, never the
        # form. Audio is resolved in media/, the .srt is written to subtitles/.
        full = dict(params, hf_token=config.HF_TOKEN,
                    data_dir=config.MEDIA_DIR, out_dir=config.SUBS_DIR)
        try:
            srt_path = run_pipeline(progress=prog, **full)
            store.update_job(job_id, status="done", stage="done", progress=1.0,
                             srt=os.path.basename(srt_path))
        except _JobCancelled:
            store.append_log(job_id, "Job canceled by user.", stage="canceled")
            store.update_job(job_id, status="canceled", stage="canceled")
        except Exception as exc:  # surface the failure to the UI
            store.append_log(job_id, f"ERROR: {exc}", stage="error")
            store.update_job(job_id, status="error", error=str(exc))


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join(STATIC_DIR, "index.html"), encoding="utf-8") as f:
        return f.read()


@app.get("/api/info")
def video_info(url: str):
    if not url.strip():
        raise HTTPException(400, "Empty URL.")
    try:
        return fetch_info(url.strip())
    except Exception as exc:
        raise HTTPException(400, f"Could not fetch info: {exc}")


@app.post("/api/downloads")
def create_download(
    url: str = Form(...),
    mode: str = Form("audio"),          # "audio" | "video"
    quality: str = Form("best"),        # "best" | "1080" | "720" | ... (video only)
    entry_ids: str = Form(""),          # comma-separated subset for playlists
):
    try:
        info = fetch_info(url.strip())
    except Exception as exc:
        raise HTTPException(400, f"Could not fetch info: {exc}")

    if info["type"] == "playlist":
        entries = info["entries"]
        wanted = {e for e in entry_ids.split(",") if e}
        if wanted:
            entries = [e for e in entries if e["id"] in wanted]
        pl_title = info.get("title") or "Playlist"
        items = [{"id": e["id"], "title": e["title"], "url": e["url"], "playlist": pl_title}
                 for e in entries if e["url"]]
    else:
        items = [{"id": info["id"], "title": info["title"], "url": info["url"]}]

    if not items:
        raise HTTPException(400, "Nothing to download.")

    os.makedirs(config.MEDIA_DIR, exist_ok=True)
    dl_id = uuid.uuid4().hex[:12]
    store.create_download(dl_id, items)
    threading.Thread(target=_download_worker, args=(dl_id, items, mode, quality),
                     daemon=True).start()
    return {"download_id": dl_id, "count": len(items)}


@app.get("/api/downloads")
def list_downloads():
    """Active (not-done) downloads — lets the UI resume after a reload."""
    return {"items": store.list_active_downloads()}


@app.get("/api/downloads/{dl_id}")
def get_download(dl_id: str):
    dl = store.get_download(dl_id)
    if dl is None:
        raise HTTPException(404, "Unknown download id.")
    return dl


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(400, "No file.")
    os.makedirs(config.MEDIA_DIR, exist_ok=True)
    name = os.path.basename(file.filename)
    ext = os.path.splitext(name)[1].lower()
    if ext not in AUDIO_EXTS and ext not in VIDEO_EXTS:
        raise HTTPException(400, f"Unsupported file type: {ext or 'unknown'}")
    dest = os.path.join(config.MEDIA_DIR, name)
    with open(dest, "wb") as out:
        out.write(await file.read())
    write_meta(os.path.splitext(name)[0],
               {"title": name, "kind": "video" if ext in VIDEO_EXTS else "audio"})
    return {"name": name}


@app.get("/api/library")
def library():
    return {"items": _list_library()}


def _safe_in(base_dir, name):
    """Resolve `name` strictly inside `base_dir` (no path traversal)."""
    path = os.path.normpath(os.path.join(base_dir, os.path.basename(name)))
    if os.path.commonpath([os.path.abspath(base_dir),
                           os.path.abspath(path)]) != os.path.abspath(base_dir):
        raise HTTPException(400, "Invalid path.")
    return path


def _safe_media(name):
    return _safe_in(config.MEDIA_DIR, name)


def _safe_subs(name):
    return _safe_in(config.SUBS_DIR, name)


@app.get("/api/library/file/{name}")
def get_media(name: str):
    path = _safe_media(name)
    if not os.path.exists(path):
        raise HTTPException(404, "File not found.")
    base = os.path.splitext(os.path.basename(name))[0]
    title = read_meta(base).get("title") or base
    ext = os.path.splitext(name)[1]
    return FileResponse(path, filename=f"{title}{ext}")


@app.delete("/api/library/{name}")
def delete_media(name: str):
    path = _safe_media(name)
    if not os.path.exists(path):
        raise HTTPException(404, "File not found.")
    base = os.path.splitext(os.path.basename(name))[0]
    os.remove(path)
    # also remove sidecar meta and any generated subtitles for this file
    subs = (os.listdir(config.SUBS_DIR) if os.path.isdir(config.SUBS_DIR) else [])
    for extra in [meta_path(base)] + [
            os.path.join(config.SUBS_DIR, s) for s in subs
            if s.startswith("subtitle-") and s.endswith(f"-{base}.srt")]:
        try:
            os.remove(extra)
        except OSError:
            pass
    return {"ok": True}


@app.get("/api/srt/{name}")
def download_srt_file(name: str):
    path = _safe_subs(name)
    if not name.lower().endswith(".srt") or not os.path.exists(path):
        raise HTTPException(404, "Dataset not found.")
    return FileResponse(path, media_type="application/x-subrip",
                        filename=os.path.basename(name))


def _list_subtitles():
    d = config.SUBS_DIR
    if not os.path.isdir(d):
        return []
    out = []
    for f in os.listdir(d):
        if not f.lower().endswith(".srt"):
            continue
        lang, base = None, None
        if f.startswith("subtitle-"):
            rest = f[len("subtitle-"):-4]          # strip "subtitle-" and ".srt"
            parts = rest.split("-", 1)
            if len(parts) == 2:
                lang, base = parts[0], parts[1]
        meta = read_meta(base) if base else {}
        path = os.path.join(d, f)
        out.append({
            "name": f,
            "lang": lang,
            "title": meta.get("title") or base or f,
            "size": os.path.getsize(path),
            "mtime": os.path.getmtime(path),
        })
    out.sort(key=lambda x: x["mtime"], reverse=True)
    return out


@app.get("/api/subtitles")
def subtitles():
    return {"items": _list_subtitles()}


@app.delete("/api/subtitles/{name}")
def delete_subtitle(name: str):
    path = _safe_subs(name)
    if not name.lower().endswith(".srt") or not os.path.exists(path):
        raise HTTPException(404, "Dataset not found.")
    os.remove(path)
    return {"ok": True}


# --------------------------------------------------------------------------- #
# Dataset (SRT segments + per-segment audio clips + export)
# --------------------------------------------------------------------------- #
def _srt_lang_base(name):
    """'subtitle-EN-my-clip.srt' -> ('EN', 'my-clip'); else (None, None)."""
    if name.startswith("subtitle-") and name.endswith(".srt"):
        rest = name[len("subtitle-"):-4]
        parts = rest.split("-", 1)
        if len(parts) == 2:
            return parts[0], parts[1]
    return None, None


def _media_for_base(base):
    """Find the source media file (any audio/video ext) for a subtitle base."""
    if not base or not os.path.isdir(config.MEDIA_DIR):
        return None
    for f in os.listdir(config.MEDIA_DIR):
        stem, ext = os.path.splitext(f)
        if stem == base and ext.lower() in AUDIO_EXTS | VIDEO_EXTS:
            return f
    return None


@app.get("/api/dataset/{name}")
def dataset(name: str):
    path = _safe_subs(name)
    if not name.lower().endswith(".srt") or not os.path.exists(path):
        raise HTTPException(404, "Dataset not found.")
    lang, base = _srt_lang_base(name)
    media = _media_for_base(base)
    meta = read_meta(base) if base else {}
    segments = parse_srt(path)
    return {
        "name": name, "base": base, "lang": lang, "media": media,
        "title": meta.get("title") or base or name,
        "count": len(segments),
        "duration": (segments[-1]["end"] if segments else 0),
        "segments": segments,
    }


def _crop_clip(media_path, start, end, out_path):
    """Crop [start, end] of media_path to a 16k mono wav at out_path."""
    dur = max(0.05, float(end) - float(start))
    subprocess.run(
        ["ffmpeg", "-y", "-i", media_path, "-ss", f"{float(start):.3f}",
         "-t", f"{dur:.3f}", "-ac", "1", "-ar", "16000", out_path],
        check=True, capture_output=True)


@app.get("/api/clip")
def clip(file: str, start: float, end: float):
    media_path = _safe_media(file)
    if not os.path.exists(media_path):
        raise HTTPException(404, "Media not found.")
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    try:
        _crop_clip(media_path, max(0.0, start), end, tmp.name)
    except subprocess.CalledProcessError:
        os.remove(tmp.name)
        raise HTTPException(500, "ffmpeg failed to extract the clip.")
    return FileResponse(tmp.name, media_type="audio/wav", filename="clip.wav",
                        background=BackgroundTask(os.remove, tmp.name))


@app.get("/api/dataset/{name}/export")
def export_dataset(name: str, fmt: str = "zip"):
    path = _safe_subs(name)
    if not name.lower().endswith(".srt") or not os.path.exists(path):
        raise HTTPException(404, "Dataset not found.")
    _, base = _srt_lang_base(name)
    stem = base or os.path.splitext(name)[0]
    segments = parse_srt(path)

    if fmt == "csv":
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["id", "start", "end", "text"])
        for s in segments:
            w.writerow([s["index"], f"{s['start']:.3f}", f"{s['end']:.3f}", s["text"]])
        return Response(
            buf.getvalue(), media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{stem}-dataset.csv"'})

    # fmt == "zip": metadata.csv + a wav clip per segment
    media = _media_for_base(base)
    if not media:
        raise HTTPException(400, "Source media is no longer in the library — "
                                 "cannot cut audio clips for the dataset.")
    media_path = os.path.join(config.MEDIA_DIR, media)
    tmpdir = tempfile.mkdtemp()
    try:
        rows = []
        for s in segments:
            fname = f"seg-{s['index']:04d}.wav"
            _crop_clip(media_path, s["start"], s["end"], os.path.join(tmpdir, fname))
            rows.append((f"clips/{fname}", s["start"], s["end"], s["text"]))

        meta_buf = io.StringIO()
        w = csv.writer(meta_buf)
        w.writerow(["file", "start", "end", "text"])
        for cn, st, en, tx in rows:
            w.writerow([cn, f"{st:.3f}", f"{en:.3f}", tx])

        import zipfile
        zip_path = os.path.join(tmpdir, "_dataset.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("metadata.csv", meta_buf.getvalue())
            for cn, *_ in rows:
                z.write(os.path.join(tmpdir, os.path.basename(cn)), cn)
        return FileResponse(
            zip_path, media_type="application/zip", filename=f"{stem}-dataset.zip",
            background=BackgroundTask(shutil.rmtree, tmpdir, True))
    except subprocess.CalledProcessError:
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise HTTPException(500, "ffmpeg failed while cutting the dataset clips.")


@app.post("/api/jobs")
def create_job(
    filename: str = Form(...),
    language: str = Form(config.LANGUAGE),
    model: str = Form(config.MODEL),
    device: str = Form(config.DEVICE),
    compute_type: str = Form(config.COMPUTE_TYPE),
    batch_size: int = Form(config.BATCH_SIZE),
):
    path = _safe_media(filename)
    if not os.path.exists(path):
        raise HTTPException(404, "File is not in the library.")
    params = {
        "audio_file": os.path.basename(filename),
        "language": language, "model": model, "device": device,
        "compute_type": compute_type, "batch_size": batch_size,
    }
    base = os.path.splitext(os.path.basename(filename))[0]
    job_id = uuid.uuid4().hex[:12]
    store.create_job(job_id, file=os.path.basename(filename),
                     title=read_meta(base).get("title") or base)
    threading.Thread(target=_worker, args=(job_id, params), daemon=True).start()
    return {"job_id": job_id}


@app.get("/api/jobs")
def list_jobs():
    """Active (queued/running) transcription jobs — lets the UI resume after a reload."""
    return {"items": store.list_active_jobs()}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = store.get_job(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job id.")
    return job


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    job = store.get_job(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job id.")
    if job["status"] in ("done", "error", "canceled", "interrupted"):
        return {"ok": True, "status": job["status"]}
    # Cooperative cancellation: a running worker checks the flag at each pipeline
    # stage boundary. A still-queued job is canceled immediately (and the worker
    # short-circuits when it later acquires _run_lock).
    store.request_cancel(job_id)
    if job["status"] == "queued":
        job = store.update_job(job_id, status="canceled", stage="canceled")
    return {"ok": True, "status": job["status"]}


@app.get("/api/jobs/{job_id}/srt")
def download_job_srt(job_id: str):
    job = store.get_job(job_id)
    if job is None or not job.get("srt"):
        raise HTTPException(404, "No transcript available for this job.")
    path = os.path.join(config.SUBS_DIR, job["srt"])
    if not os.path.exists(path):
        raise HTTPException(404, "Dataset file missing on disk.")
    return FileResponse(path, media_type="application/x-subrip", filename=job["srt"])


@app.get("/api/gpu")
def gpu():
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode()
        return {"available": True, "name": name, "util": util.gpu,
                "mem_used": mem.used // 2**20, "mem_total": mem.total // 2**20}
    except Exception as exc:
        return {"available": False, "error": str(exc)}


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
