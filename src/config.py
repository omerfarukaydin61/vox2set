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
# Job/download state lives in an in-memory registry (web/jobs.py), not on disk,
# so it does NOT survive a process restart (the library/datasets on disk do).
DATA_DIR = os.getenv("DATA_DIR", "data")
MEDIA_DIR = os.path.join(DATA_DIR, "media")
SUBS_DIR = os.path.join(DATA_DIR, "subtitles")

# Codec used when extracting audio from a downloaded video.
AUDIO_CODEC = os.getenv("AUDIO_CODEC", "mp3")

# --- Job registry / worker pool ---------------------------------------------
# Each job runs in its own subprocess (bulkhead isolation). These cap how many
# of each kind run at once; one GPU pipeline + one download by default, and the
# two overlap (network/ffmpeg vs GPU). The registry keeps at most JOB_HISTORY
# terminal jobs before evicting the oldest (so memory doesn't grow forever).
MAX_TRANSCRIBE_JOBS = int(os.getenv("MAX_TRANSCRIBE_JOBS", "1"))
MAX_DOWNLOAD_JOBS = int(os.getenv("MAX_DOWNLOAD_JOBS", "1"))
JOB_HISTORY = int(os.getenv("JOB_HISTORY", "200"))

# --- Model / hardware --------------------------------------------------------
MODEL = os.getenv("MODEL", "large-v3")
DEVICE = os.getenv("DEVICE", "cuda")          # "cuda" or "cpu"
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "float16")  # "int8" if low on memory
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))      # reduce if low on GPU memory

# LANGUAGE: the spoken language of the source audio (NOT a translation target;
# Whisper transcribes in the original language). "auto"/blank => auto-detect.
# The UI defaults the picker to Auto-detect but lets the user force a language
# (auto-detect can misfire, e.g. English intro music read as Chinese).
LANGUAGE = os.getenv("LANGUAGE", "auto")

# Local offline translation model (HuggingFace) for the dataset modal's
# "Translate subtitles to…" feature. NLLB-200 distilled-600M covers 200 langs
# and runs on the same GPU; no API key / per-character cost.
TRANSLATE_MODEL = os.getenv("TRANSLATE_MODEL", "facebook/nllb-200-distilled-600M")
