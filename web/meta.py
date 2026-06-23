"""Sidecar metadata helpers ({base}.meta.json next to each media file).

Used by web/app.py (downloads write a sidecar; library/dataset reads read it);
kept in its own module so it stays import-light and reusable.
"""
import json
import os

from src import config


def meta_path(base):
    # Sidecars live next to their media file, under data/media/.
    return os.path.join(config.MEDIA_DIR, f"{base}.meta.json")


def write_meta(base, meta):
    try:
        os.makedirs(config.MEDIA_DIR, exist_ok=True)
        with open(meta_path(base), "w", encoding="utf-8") as f:
            json.dump(meta, f)
    except OSError:
        pass


def read_meta(base):
    try:
        with open(meta_path(base), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}
