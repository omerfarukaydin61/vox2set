"""FastAPI web interface for VoxSet — a speech dataset builder.

Two decoupled stages, each run as a subprocess per job (process-per-job
bulkhead) supervised from this single API process:
  * Downloads     — fetch a single video or a whole playlist, as audio or video,
                    into the data/ library (one download job at a time).
  * Transcription — pick a file from the library and run the whisperx pipeline
                    (transcribe/align/diarize) into a speaker-labeled speech
                    dataset, stored as an .srt (one GPU job at a time).

Job state lives in an in-memory registry (web/jobs.py): a finite state machine
per job, a ring buffer of log lines, and a pub/sub stream. Clients tail it live
over Server-Sent Events (/api/jobs/{id}/events, /api/downloads/{id}/events); the
list endpoints let the UI re-attach after a page reload. State is NOT persisted,
so a *process* restart starts empty (the media library and datasets on disk are
unaffected); a browser reload re-attaches via the registry.
"""
import csv
import io
import json
import os
import queue as _q
import shutil
import subprocess
import tempfile

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTask

from src import config
from src.downloader import fetch_info
from src.util import parse_srt, read_speakers, speakers_path
from web import jobs, runner
from web.meta import meta_path, read_meta, write_meta

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(HERE, "static")

app = FastAPI(title="VoxSet")

AUDIO_EXTS = {".mp3", ".m4a", ".wav", ".opus", ".flac", ".aac", ".ogg"}
VIDEO_EXTS = {".mp4", ".mkv", ".webm", ".mov", ".avi"}


def _migrate_flat_layout():
    """Move any pre-existing flat data/ files into media/ and subtitles/.

    Older versions dumped media, .meta.json sidecars and subtitle-*.srt all in
    the data/ root. This relocates them once so an upgrade doesn't "lose" the
    library. .gitkeep and stray *.part files stay at the root.
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
    """Tidy the data layout on startup. Job state is in-memory, so there is no
    persisted work to reconcile — a fresh process simply starts with no jobs."""
    try:
        _migrate_flat_layout()
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
# Server-Sent Events: snapshot + live tail of one job
# --------------------------------------------------------------------------- #
def _sse_stream(jid):
    """Yield SSE frames: the current snapshot, then every subsequent snapshot,
    until the job reaches a terminal state. A periodic comment keeps the
    connection from idling out."""
    snap, q = jobs.subscribe(jid)
    if snap is None:
        return
    try:
        yield f"data: {json.dumps(snap)}\n\n"
        if snap.get("status") in jobs.TERMINAL:
            return
        while True:
            try:
                snap = q.get(timeout=15)
            except _q.Empty:
                yield ": keep-alive\n\n"
                continue
            yield f"data: {json.dumps(snap)}\n\n"
            if snap.get("status") in jobs.TERMINAL:
                return
    finally:
        jobs.unsubscribe(jid, q)


_SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


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
    dl_id = runner.submit_download(items, mode, quality)
    return {"download_id": dl_id, "count": len(items)}


def _get_download_or_404(dl_id):
    dl = jobs.get(dl_id)
    if dl is None or dl.get("kind") != "download":
        raise HTTPException(404, "Unknown download id.")
    return dl


@app.get("/api/downloads")
def list_downloads():
    """Active downloads — lets the UI re-attach after a reload."""
    return {"items": [{"id": j["id"], "status": j["status"]}
                      for j in jobs.list_active("download")]}


@app.get("/api/downloads/{dl_id}")
def get_download(dl_id: str):
    return _get_download_or_404(dl_id)


@app.get("/api/downloads/{dl_id}/events")
def download_events(dl_id: str):
    _get_download_or_404(dl_id)
    return StreamingResponse(_sse_stream(dl_id), media_type="text/event-stream",
                             headers=_SSE_HEADERS)


@app.post("/api/downloads/{dl_id}/cancel")
def cancel_download(dl_id: str):
    _get_download_or_404(dl_id)
    status = runner.cancel_download(dl_id)
    return {"ok": True, "status": status}


@app.post("/api/downloads/{dl_id}/cancel/{idx}")
def cancel_download_item(dl_id: str, idx: int):
    dl = _get_download_or_404(dl_id)
    if not (0 <= idx < len(dl.get("items", []))):
        raise HTTPException(404, "Unknown item index.")
    # Cancel just this entry; the rest of the playlist keeps going. A pending
    # item is marked canceled immediately, a downloading one aborts at its next
    # yt-dlp progress tick.
    runner.cancel_download_item(dl_id, idx)
    return {"ok": True}


@app.post("/api/downloads/{dl_id}/retry/{idx}")
def retry_download_item(dl_id: str, idx: int):
    dl = _get_download_or_404(dl_id)
    if not (0 <= idx < len(dl.get("items", []))):
        raise HTTPException(404, "Unknown item index.")
    if dl["items"][idx]["status"] != "error":
        raise HTTPException(400, "Only failed items can be retried.")
    if not dl["items"][idx].get("url"):
        raise HTTPException(400, "No saved URL to retry this item.")
    runner.retry_download_item(dl_id, idx)
    return {"ok": True}


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
    # also remove sidecar meta and any generated subtitles (+ their speaker
    # sidecars) for this file
    subs = (os.listdir(config.SUBS_DIR) if os.path.isdir(config.SUBS_DIR) else [])
    srt_paths = [os.path.join(config.SUBS_DIR, s) for s in subs
                 if s.startswith("subtitle-") and s.endswith(f"-{base}.srt")]
    for extra in [meta_path(base)] + srt_paths + [speakers_path(p) for p in srt_paths]:
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
            "base": base,
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
    try:
        os.remove(speakers_path(path))     # drop the speaker sidecar too
    except OSError:
        pass
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
    spk = read_speakers(path)
    for i, s in enumerate(segments):
        s["speaker"] = spk[i] if i < len(spk) else None
    return {
        "name": name, "base": base, "lang": lang, "media": media,
        "title": meta.get("title") or base or name,
        "count": len(segments),
        "duration": (segments[-1]["end"] if segments else 0),
        "speakers": sorted({s for s in spk if s}),
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
    spk = read_speakers(path)
    speaker = lambda i: (spk[i] if i < len(spk) else "") or ""

    if fmt == "csv":
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["id", "speaker", "start", "end", "text"])
        for i, s in enumerate(segments):
            w.writerow([s["index"], speaker(i), f"{s['start']:.3f}", f"{s['end']:.3f}", s["text"]])
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
        for i, s in enumerate(segments):
            fname = f"seg-{s['index']:04d}.wav"
            _crop_clip(media_path, s["start"], s["end"], os.path.join(tmpdir, fname))
            rows.append((f"clips/{fname}", speaker(i), s["start"], s["end"], s["text"]))

        meta_buf = io.StringIO()
        w = csv.writer(meta_buf)
        w.writerow(["file", "speaker", "start", "end", "text"])
        for cn, sp, st, en, tx in rows:
            w.writerow([cn, sp, f"{st:.3f}", f"{en:.3f}", tx])

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


# --------------------------------------------------------------------------- #
# Translate a dataset's .srt into another language (sentence-merge → NLLB →
# reflow). Driven entirely from the dataset modal; tailed over SSE there.
# --------------------------------------------------------------------------- #
@app.post("/api/dataset/{name}/translate")
def translate_dataset(name: str, to: str = Form(...)):
    from src.translate import FLORES
    path = _safe_subs(name)
    if not name.lower().endswith(".srt") or not os.path.exists(path):
        raise HTTPException(404, "Dataset not found.")
    lang, base = _srt_lang_base(name)
    src = (lang or "").lower()
    target = to.strip().lower()
    if not src:
        raise HTTPException(400, "Source language is unknown for this dataset.")
    if not target:
        raise HTTPException(400, "No target language.")
    if src == target:
        raise HTTPException(400, "Source and target language are the same.")
    if src not in FLORES:
        raise HTTPException(400, f"Translation source language not supported: {src}")
    if target not in FLORES:
        raise HTTPException(400, f"Translation target language not supported: {target}")
    title = (read_meta(base).get("title") if base else None) or base
    job_id = runner.submit_translate(name, base, src, target, title=title)
    return {"job_id": job_id}


@app.get("/api/translate")
def list_translations():
    """Active translation jobs — lets the UI re-attach after a page reload."""
    return {"items": [{"id": j["id"], "base": (j.get("params") or {}).get("base"),
                       "target": j.get("target"), "status": j["status"],
                       "progress": j.get("progress", 0.0)}
                      for j in jobs.list_active("translate")]}


def _get_translate_or_404(job_id):
    j = jobs.get(job_id)
    if j is None or j.get("kind") != "translate":
        raise HTTPException(404, "Unknown translate job.")
    return j


@app.get("/api/translate/{job_id}")
def get_translate(job_id: str):
    return _get_translate_or_404(job_id)


@app.get("/api/translate/{job_id}/events")
def translate_events(job_id: str):
    _get_translate_or_404(job_id)
    return StreamingResponse(_sse_stream(job_id), media_type="text/event-stream",
                             headers=_SSE_HEADERS)


@app.post("/api/translate/{job_id}/cancel")
def cancel_translate(job_id: str):
    _get_translate_or_404(job_id)
    runner.cancel_translate(job_id)
    return {"ok": True}


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
    title = read_meta(base).get("title") or base
    job_id = runner.submit_transcribe(os.path.basename(filename), title, params)
    return {"job_id": job_id}


def _get_job_or_404(job_id):
    job = jobs.get(job_id)
    if job is None or job.get("kind") != "transcribe":
        raise HTTPException(404, "Unknown job id.")
    return job


@app.get("/api/jobs")
def list_jobs():
    """Active (queued/running) transcription jobs — lets the UI re-attach after a reload."""
    return {"items": [{"id": j["id"], "file": j.get("file"),
                       "title": j.get("title") or j.get("file"),
                       "status": j["status"], "stage": j.get("stage"),
                       "progress": j.get("progress", 0.0)}
                      for j in jobs.list_active("transcribe")]}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    return _get_job_or_404(job_id)


@app.get("/api/jobs/{job_id}/events")
def job_events(job_id: str):
    _get_job_or_404(job_id)
    return StreamingResponse(_sse_stream(job_id), media_type="text/event-stream",
                             headers=_SSE_HEADERS)


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    _get_job_or_404(job_id)
    # transcribe cancel terminates the subprocess (frees the GPU immediately);
    # a still-queued job is marked canceled before it ever spawns.
    status = runner.cancel_transcribe(job_id)
    return {"ok": True, "status": status}


@app.post("/api/jobs/{job_id}/retry")
def retry_job(job_id: str):
    job = _get_job_or_404(job_id)
    if job["status"] not in ("error", "interrupted", "canceled"):
        raise HTTPException(400, "Only failed jobs can be retried.")
    params = job.get("params")
    if not params or not params.get("audio_file"):
        raise HTTPException(400, "No saved settings to retry this job.")
    if not os.path.exists(_safe_media(params["audio_file"])):
        raise HTTPException(404, "Source file is no longer in the library.")
    # reuse the same id so the existing card revives in place
    runner.retry_transcribe(job_id)
    return {"ok": True}


@app.get("/api/jobs/{job_id}/srt")
def download_job_srt(job_id: str):
    job = _get_job_or_404(job_id)
    if not job.get("srt"):
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
