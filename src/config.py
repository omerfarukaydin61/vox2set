# config.py
# All runtime settings come from environment variables (with sensible defaults),
# so the same image can be driven entirely from docker-compose / .env.
from dotenv import load_dotenv
import os

load_dotenv()

# --- Secrets -----------------------------------------------------------------
HF_TOKEN = os.getenv("HF_TOKEN")  # required for diarization

# --- I/O ---------------------------------------------------------------------
# Root data directory. In Docker this is the bind-mounted ./data volume. It is
# kept tidy with sub-directories instead of one flat dump:
#   media/      source media (downloaded + uploaded) + their .meta.json sidecars
#   subtitles/  generated subtitle-*.srt datasets
#   voxset.db   SQLite job state (survives a restart; see web/store.py)
DATA_DIR = os.getenv("DATA_DIR", "data")
MEDIA_DIR = os.path.join(DATA_DIR, "media")
SUBS_DIR = os.path.join(DATA_DIR, "subtitles")

# Codec used when extracting audio from a downloaded video.
AUDIO_CODEC = os.getenv("AUDIO_CODEC", "mp3")

# --- Model / hardware --------------------------------------------------------
MODEL = os.getenv("MODEL", "large-v3")
DEVICE = os.getenv("DEVICE", "cuda")          # "cuda" or "cpu"
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "float16")  # "int8" if low on memory
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))      # reduce if low on GPU memory

# LANGUAGE: target language code for the subtitles (e.g. "en", "tr").
LANGUAGE = os.getenv("LANGUAGE", "en")
