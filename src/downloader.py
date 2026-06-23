import os
import glob
from .util import log as default_log


def thumb_for(entry):
    """Best-effort thumbnail URL from a yt-dlp entry/info dict."""
    if entry.get("thumbnail"):
        return entry["thumbnail"]
    thumbs = entry.get("thumbnails") or []
    if thumbs:
        return thumbs[-1].get("url")
    vid = entry.get("id")
    return f"https://i.ytimg.com/vi/{vid}/hqdefault.jpg" if vid else None


def _entry_url(e):
    return (e.get("url") or e.get("webpage_url")
            or (f"https://www.youtube.com/watch?v={e['id']}" if e.get("id") else None))


# Playlists routinely contain private/deleted/members-only videos. Flat extraction
# still lists them (often with no title — just the id — or a bracket placeholder),
# but they can't be downloaded without account cookies, so they'd only ever queue
# up and fail. Drop them from the preview so they never enter the download list.
_UNAVAILABLE_TITLES = {"[private video]", "[deleted video]", "[unavailable video]"}
_UNAVAILABLE_AVAIL = {"private", "needs_auth", "premium_only", "subscriber_only"}


def _is_unavailable(e):
    title = (e.get("title") or "").strip()
    if not title:  # flat extraction returns no title for private/deleted videos
        return True
    if title.lower() in _UNAVAILABLE_TITLES:
        return True
    if (e.get("availability") or "").lower() in _UNAVAILABLE_AVAIL:
        return True
    return False


def fetch_info(url):
    """Return normalized metadata for `url` without downloading.

    Single video -> {type:"video", id, title, uploader, duration, thumbnail, url}
    Playlist     -> {type:"playlist", title, uploader, count, entries:[...]}
    Playlist entries are extracted *flat* (fast; no per-video format probing).
    """
    import yt_dlp  # imported lazily so the server starts without it installed

    opts = {"quiet": True, "skip_download": True, "extract_flat": "in_playlist"}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)

    if info.get("_type") == "playlist":
        entries = []
        for e in info.get("entries") or []:
            if not e or not e.get("id") or _is_unavailable(e):
                continue
            entries.append({
                "id": e.get("id"),
                "title": e.get("title") or e.get("id"),
                "duration": e.get("duration"),
                "thumbnail": thumb_for(e),
                "url": _entry_url(e),
            })
        return {
            "type": "playlist",
            "title": info.get("title"),
            "uploader": info.get("uploader") or info.get("channel"),
            "count": len(entries),
            "entries": entries,
        }

    return {
        "type": "video",
        "id": info.get("id"),
        "title": info.get("title"),
        "uploader": info.get("uploader") or info.get("channel"),
        "duration": info.get("duration"),
        "thumbnail": thumb_for(info),
        "url": info.get("webpage_url") or url,
    }


def _format_selector(mode, quality):
    """yt-dlp format string for an audio-only or video download."""
    if mode == "video":
        if not quality or quality == "best":
            return "bestvideo*+bestaudio/best"
        return (f"bestvideo[height<={quality}]+bestaudio/best[height<={quality}]/"
                f"best[height<={quality}]/best")
    return "bestaudio/best"


def _resolve_output(out_dir, vid, mode, codec, info):
    """Find the final file on disk for video id `vid`."""
    if mode == "audio":
        cand = os.path.join(out_dir, f"{vid}.{codec}")
        if os.path.exists(cand):
            return cand
    reqs = info.get("requested_downloads")
    if reqs and reqs[0].get("filepath") and os.path.exists(reqs[0]["filepath"]):
        return reqs[0]["filepath"]
    for p in sorted(glob.glob(os.path.join(out_dir, f"{vid}.*"))):
        if not p.endswith((".info.json", ".meta.json", ".part", ".ytdl")):
            return p
    return None


def download_media(url, out_dir, mode="audio", quality="best", codec="mp3",
                   log=default_log, progress_hook=None):
    """Download one video as audio (extracted to `codec`) or as a merged mp4 video.

    Returns (path, info) where info is the yt-dlp metadata dict. Requires ffmpeg.
    `progress_hook` is an optional yt-dlp progress hook.
    """
    import yt_dlp  # imported lazily so the web server starts without it installed

    os.makedirs(out_dir, exist_ok=True)
    ydl_opts = {
        "format": _format_selector(mode, quality),
        # %(id)s gives a stable, filesystem-safe basename we can reconstruct.
        "outtmpl": os.path.join(out_dir, "%(id)s.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }
    if mode == "audio":
        ydl_opts["postprocessors"] = [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": codec,
            "preferredquality": "192",
        }]
    else:
        ydl_opts["merge_output_format"] = "mp4"
    if progress_hook:
        ydl_opts["progress_hooks"] = [progress_hook]

    log(f"Downloading {mode} from: {url}")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)

    path = _resolve_output(out_dir, info["id"], mode, codec, info)
    if not path:
        raise RuntimeError(f"Download finished but output file not found for {info.get('id')}.")
    log(f"Downloaded to: {path}")
    return path, info
