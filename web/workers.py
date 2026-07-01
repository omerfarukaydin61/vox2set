"""Child-process entry points for the job runner.

Each job runs in its own spawned subprocess (process-per-job bulkhead): a crash
can't take down the API, and finishing the process releases its VRAM/RAM
cleanly. These functions are the subprocess targets — they talk to the parent
ONLY through plain multiprocessing queues carrying picklable dicts.

Keep this module's import surface light: it is re-imported in every spawned
child. Heavy deps (torch/whisperx via src.pipeline, yt-dlp via src.downloader)
are imported lazily inside the functions, not at module load.
"""
import os
import queue as _q


class _Cancelled(Exception):
    """Raised inside the yt-dlp hook to abort the current download item."""


def _emit(events, **ev):
    try:
        events.put(ev)
    except Exception:
        pass


def run_transcribe(events, params):
    """Run the whisperx pipeline, emitting log/done/error events.

    Cancellation is performed by the parent terminating this process (which
    frees the GPU immediately), so there is no cooperative cancel check here.
    """
    from src import config
    from src.pipeline import run_pipeline

    def prog(stage, msg):
        _emit(events, t="log", stage=stage, msg=msg)

    full = dict(params, hf_token=config.HF_TOKEN,
                data_dir=config.MEDIA_DIR, out_dir=config.SUBS_DIR)
    try:
        srt_path = run_pipeline(progress=prog, **full)
        _emit(events, t="done", srt=os.path.basename(srt_path))
    except Exception as exc:
        _emit(events, t="error", error=str(exc))


def run_translate(events, params):
    """Translate an existing .srt to another language (sentence-merge → NLLB →
    reflow) and write `subtitle-{TARGET}-{base}.srt`. Cancellation = parent
    terminates this process."""
    import os
    from src import config
    from src.util import parse_srt, write_srt, write_speakers, read_speakers
    from src import translate as T

    def prog(stage, frac, msg):
        ev = {"t": "log", "stage": stage}
        if msg:
            ev["msg"] = msg
        if frac is not None:
            ev["progress"] = frac
        _emit(events, **ev)

    try:
        src_path = os.path.join(config.SUBS_DIR, params["srt_name"])
        segs = parse_srt(src_path)
        spk = read_speakers(src_path)                 # carry diarization labels over
        for i, s in enumerate(segs):
            s["speaker"] = spk[i] if i < len(spk) else None
        cues = T.translate_srt(segs, params["src"], params["target"],
                               model_name=config.TRANSLATE_MODEL, progress=prog)
        out_path = os.path.join(config.SUBS_DIR,
                                f"subtitle-{params['target'].upper()}-{params['base']}.srt")
        write_srt(cues, out_path)
        write_speakers(out_path, cues)
        _emit(events, t="done", srt=os.path.basename(out_path))
    except Exception as exc:
        _emit(events, t="error", error=str(exc))


def run_download(events, control, items, mode, quality):
    """Download `items` (list of (idx, item_dict)) one by one, emitting per-item
    events. Cooperative cancellation: the parent puts `idx` (cancel one item) or
    "all" (cancel the rest) on the `control` queue; we drain it before each item
    and on every yt-dlp progress tick.
    """
    from src import config
    from src.downloader import download_media, thumb_for
    from web.meta import write_meta

    cancel_all = False
    canceled = set()

    def drain():
        nonlocal cancel_all
        while True:
            try:
                msg = control.get_nowait()
            except _q.Empty:
                break
            if msg == "all":
                cancel_all = True
            elif isinstance(msg, int):
                canceled.add(msg)

    for idx, it in items:
        drain()
        if cancel_all or idx in canceled:
            _emit(events, t="item", idx=idx, status="canceled")
            continue

        _emit(events, t="item", idx=idx, status="downloading", progress=0.0)
        last = {"pct": -1}

        def hook(d, idx=idx, last=last):
            drain()
            if cancel_all or idx in canceled:
                raise _Cancelled()
            if d.get("status") == "downloading":
                total = d.get("total_bytes") or d.get("total_bytes_estimate")
                if total:
                    frac = d.get("downloaded_bytes", 0) / total
                    pct = int(frac * 100)
                    if pct != last["pct"]:        # throttle event volume
                        last["pct"] = pct
                        _emit(events, t="item", idx=idx, progress=frac)
            elif d.get("status") == "finished":
                _emit(events, t="item", idx=idx, progress=1.0, status="processing")

        try:
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
            _emit(events, t="item", idx=idx, status="done", progress=1.0,
                  file=os.path.basename(path))
        except _Cancelled:
            _emit(events, t="item", idx=idx, status="canceled")
        except Exception as exc:
            drain()
            if cancel_all or idx in canceled:
                _emit(events, t="item", idx=idx, status="canceled")
            else:
                _emit(events, t="item", idx=idx, status="error", error=str(exc))

    _emit(events, t="finished", cancel_all=cancel_all)
