import os
import re
import time

def log(msg):
    print(f"[LOG] {time.strftime('%Y-%m-%d %H:%M:%S')} - {msg}")

def format_timestamp(seconds):
    ms = int((seconds - int(seconds)) * 1000)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

def srt_path_for(audio_path, language, out_dir):
    base = os.path.splitext(os.path.basename(audio_path))[0]
    return os.path.join(out_dir, f"subtitle-{language.upper()}-{base}.srt")

def write_srt(segments, srt_path):
    with open(srt_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            start = format_timestamp(seg["start"])
            end = format_timestamp(seg["end"])
            text = seg.get("text", "")
            f.write(f"{i}\n{start} --> {end}\n{text}\n\n")


def _ts_to_seconds(ts):
    """'00:01:02,500' (or with a dot) -> 62.5 seconds."""
    ts = ts.strip().replace(",", ".")
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def parse_srt(path):
    """Parse an .srt file into [{index, start, end, text}] (seconds as floats)."""
    with open(path, encoding="utf-8") as f:
        content = f.read()
    segments = []
    for block in re.split(r"\n\s*\n", content.strip()):
        lines = [ln for ln in block.splitlines() if ln.strip()]
        tc_idx = next((i for i, ln in enumerate(lines) if "-->" in ln), None)
        if tc_idx is None:
            continue
        try:
            start_raw, end_raw = lines[tc_idx].split("-->")
            start, end = _ts_to_seconds(start_raw), _ts_to_seconds(end_raw)
        except (ValueError, IndexError):
            continue
        text = " ".join(lines[tc_idx + 1:]).strip()
        segments.append({"index": len(segments) + 1,
                         "start": start, "end": end, "text": text})
    return segments
