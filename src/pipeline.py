import os
import warnings
from importlib.metadata import version

from .util import log as default_log, write_srt, write_speakers, srt_path_for


def _free_cuda():
    """Release cached GPU memory (between models / before an OOM retry)."""
    try:
        import gc
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
    except Exception:
        pass


def _is_oom(exc):
    """True for CUDA/GPU out-of-memory errors (torch *or* ctranslate2)."""
    return ("out of memory" in str(exc).lower()
            or exc.__class__.__name__ == "OutOfMemoryError")


def _gpu_free_gb():
    """Free VRAM (GiB) on the active CUDA device, or None if it can't be read."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        free, _total = torch.cuda.mem_get_info()
        return free / (1024 ** 3)
    except Exception:
        return None


def _auto_batch_size(device, compute_type):
    """Pick a conservative *starting* batch_size from free VRAM.

    This only avoids wasting a load+OOM cycle on the first try; the transcribe
    loop still halves batch_size on any out-of-memory error, so a wrong guess
    self-corrects. Deliberately conservative — overshooting costs a retry,
    undershooting only costs a little speed. Best measured *after* the ASR model
    is loaded, so the model's own weights are already subtracted from `free`.
    """
    if (device or "").lower() != "cuda":
        return 4  # CPU: RAM is plentiful but batching barely helps throughput
    free = _gpu_free_gb()
    if free is None:
        return 8  # unknown VRAM — middle of the road; the backoff covers us
    # int8 roughly halves the footprint vs float16/32 → allow ~1.6x more.
    if (compute_type or "").lower() == "int8":
        free *= 1.6
    for gb, bs in ((20, 24), (14, 16), (10, 12), (7, 8), (5, 4), (3.5, 2)):
        if free >= gb:
            return bs
    return 1


def _resolve_audio(audio_file, data_dir):
    """Resolve a local audio path. A bare filename is taken inside data_dir; an
    absolute or path-containing value is used as-is."""
    if not audio_file:
        raise ValueError("No audio file given to run_pipeline().")
    if os.path.isabs(audio_file) or os.path.dirname(audio_file):
        return audio_file
    return os.path.join(data_dir, audio_file)


def _patch_whisperx_vad_for_pyannote4():
    """Make whisperx 3.3.4's bundled VAD work with pyannote.audio 4.x.

    whisperx.load_model() builds a Voice-Activity-Detection pipeline and calls it
    with the old `use_auth_token=` kwarg. pyannote 4.x renamed that to `token=`,
    so the value leaks through **inference_kwargs into Inference.__init__ and
    raises "Inference.__init__() got an unexpected keyword argument
    'use_auth_token'" — during load_model, before transcription even starts.

    whisperx's VoiceActivitySegmentation.__init__ only forwards to its pyannote
    parent, so we replace it with a version that maps use_auth_token -> token.
    """
    from whisperx.vads import pyannote as wvp
    from pyannote.audio.pipelines.voice_activity_detection import VoiceActivityDetection

    def __init__(self, segmentation="pyannote/segmentation", fscore=False,
                 use_auth_token=None, **inference_kwargs):
        VoiceActivityDetection.__init__(
            self, segmentation=segmentation, fscore=fscore,
            token=use_auth_token, **inference_kwargs)

    wvp.VoiceActivitySegmentation.__init__ = __init__


def _diarize(audio, hf_token, device, model_name="pyannote/speaker-diarization-3.1"):
    """Speaker diarization via pyannote.audio 4.x.

    whisperx 3.3.4's bundled `DiarizationPipeline` calls pyannote with the old
    `use_auth_token=` kwarg, which pyannote 4.x removed (renamed to `token=`),
    raising "Inference.__init__() got an unexpected keyword argument
    'use_auth_token'". We can't downgrade pyannote because the 3.x line needs an
    old torchaudio that's incompatible with the cu128 torch required for Blackwell.

    So we drive pyannote 4.x directly. The audio is passed as an in-memory
    waveform dict (whisperx already decoded it with ffmpeg) — this also sidesteps
    torchcodec, whose file decoding fails on the CUDA 12.8 base image.

    Returns a DataFrame with start/end/speaker columns for assign_word_speakers.
    """
    import torch
    import pandas as pd
    from pyannote.audio import Pipeline

    SAMPLE_RATE = 16000  # whisperx.load_audio resamples to 16 kHz mono

    try:
        pipeline = Pipeline.from_pretrained(model_name, token=hf_token)
    except Exception as exc:
        raise RuntimeError(
            f"Could not download the diarization model '{model_name}'. This is "
            "almost always a Hugging Face access problem, not a bug. Make sure you "
            "have:\n"
            "  1. Accepted the user conditions at huggingface.co/pyannote/"
            "speaker-diarization-3.1 AND huggingface.co/pyannote/segmentation-3.0\n"
            "  2. Set HF_TOKEN to a token with 'Read access to contents of all "
            "public gated repos' enabled (fine-grained tokens disable this by "
            f"default).\nOriginal error: {exc}"
        ) from exc
    if pipeline is None:
        raise RuntimeError(
            f"Could not load '{model_name}'. Check HF_TOKEN and that you accepted "
            "the model's user conditions on Hugging Face."
        )
    pipeline.to(torch.device(device))

    waveform = torch.from_numpy(audio[None, :])  # (channel, time)
    output = pipeline({"waveform": waveform, "sample_rate": SAMPLE_RATE})

    # pyannote 4.x may return a wrapper object; the legacy 3.1 pipeline returns a
    # plain Annotation. Handle both.
    annotation = getattr(output, "speaker_diarization", output)

    df = pd.DataFrame(annotation.itertracks(yield_label=True),
                      columns=["segment", "label", "speaker"])
    df["start"] = df["segment"].apply(lambda s: s.start)
    df["end"] = df["segment"].apply(lambda s: s.end)
    return df


def _tighten_segments(segments, pad=0.15):
    """Trim each cue to its actual word timestamps so it disappears when the
    speech ends instead of lingering through the silence until the next cue.

    Whisper's segment `end` is often stretched to (or near) the next segment's
    start; alignment gives per-word times, so we reset start/end from the first
    and last *timed* words (+ a small trailing pad), then clamp so a cue never
    overlaps the next. No-op for segments without word timings (alignment
    unavailable), which simply keep Whisper's coarse boundaries.
    """
    for seg in segments:
        words = [w for w in seg.get("words", [])
                 if w.get("start") is not None and w.get("end") is not None]
        if not words:
            continue
        seg["start"] = max(0.0, words[0]["start"])
        seg["end"] = words[-1]["end"] + pad
    for cur, nxt in zip(segments, segments[1:]):
        if cur["end"] > nxt["start"]:          # don't bleed into the next cue
            cur["end"] = nxt["start"]
    return segments


def run_pipeline(*, audio_file=None, language="en",
                 model="large-v3", device="cuda", compute_type="float16",
                 batch_size=16, hf_token=None, data_dir="data", out_dir=None,
                 progress=None):
    """Transcribe + align + diarize a local audio file into a speech dataset (.srt).

    `data_dir` is where a bare `audio_file` name is resolved (the media dir).
    `out_dir` is where the .srt is written; defaults to `data_dir` when omitted.
    `progress` is an optional callable(stage: str, message: str) for UIs.
    Returns the path to the generated dataset (.srt) file.
    """
    def log(msg, stage="info"):
        default_log(msg)
        if progress:
            progress(stage, msg)

    import whisperx  # imported lazily so the web server starts without GPU libs

    if version("whisperx") != "3.3.4":
        warnings.warn(f"Recommended to use whisperx==3.3.4, but found version {version('whisperx')}")

    _patch_whisperx_vad_for_pyannote4()

    audio_path = _resolve_audio(audio_file, data_dir)
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    _free_cuda()  # reclaim VRAM left over from a previous (serialized) job

    log(f"Loading model '{model}' on device: {device} (compute_type={compute_type})", "load")
    asr_model = whisperx.load_model(model, device, compute_type=compute_type)

    log(f"Loading audio file: {audio_path}", "load")
    audio = whisperx.load_audio(audio_path)

    # blank / "auto" -> let Whisper detect the language
    lang = None if (language or "").strip().lower() in ("", "auto") else language

    # Transcribe with automatic batch-size backoff on CUDA out-of-memory: halve
    # the batch and free VRAM until it fits (down to 1) instead of failing.
    # batch_size <= 0 means "auto": pick a starting point from free VRAM now that
    # the ASR model is loaded; the backoff loop below still protects us.
    if int(batch_size) <= 0:
        bs = _auto_batch_size(device, compute_type)
        log(f"Auto-selected batch_size={bs} from available VRAM.", "transcribe")
    else:
        bs = max(1, int(batch_size))
    while True:
        log(f"Transcribing audio (batch_size={bs}, language={lang or 'auto-detect'})", "transcribe")
        try:
            result = asr_model.transcribe(audio, batch_size=bs, language=lang)
            break
        except Exception as exc:
            if _is_oom(exc) and bs > 1:
                bs = max(1, bs // 2)
                log(f"GPU out of memory — freeing memory and retrying at batch_size={bs}…", "transcribe")
                _free_cuda()
                continue
            raise
    detected = result.get("language") or lang or "und"
    log(f"Transcription complete (language={detected}, batch_size={bs}).", "transcribe")

    # free the ASR model before loading the align model (lowers peak VRAM)
    del asr_model
    _free_cuda()

    # name the SRT after the effective language (so auto-detect names it correctly)
    srt_dir = out_dir or data_dir
    os.makedirs(srt_dir, exist_ok=True)
    srt_path = srt_path_for(audio_path, detected, srt_dir)

    # Alignment + diarization are best-effort: whisperx only ships wav2vec align
    # models for a subset of languages, so for the rest we keep segment-level
    # timestamps instead of failing the whole job.
    log("Loading alignment model and aligning segments...", "align")
    aligned = False
    try:
        model_a, metadata = whisperx.load_align_model(language_code=detected, device=device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
        aligned = True
        del model_a, metadata
        _free_cuda()
        log("Alignment complete.", "align")
    except Exception as exc:  # no align model for this language (or it failed to load)
        log(f"Alignment unavailable for '{detected}' ({exc}); using segment-level timestamps.", "align")

    if aligned:
        log("Running diarization...", "diarize")
        try:
            diarize_segments = _diarize(audio, hf_token, device)
            result = whisperx.assign_word_speakers(diarize_segments, result)
            log("Diarization and speaker assignment complete.", "diarize")
        except Exception as exc:  # HF token / gated repo / network — don't lose the whole dataset
            log(f"Diarization failed ({exc}); writing dataset WITHOUT speaker labels. "
                f"Check HF_TOKEN and gated-repo access if you need speakers.", "diarize")
    else:
        log("Skipping diarization (needs word-level alignment).", "diarize")

    _free_cuda()
    log(f"Writing dataset to: {srt_path}", "write")
    # tighten cue end times to the actual speech (when aligned) so subtitles
    # don't hang on screen through the silence before the next line
    _tighten_segments(result["segments"])
    write_srt(result["segments"], srt_path)
    write_speakers(srt_path, result["segments"])   # speaker labels sidecar (if any)
    log(f"Dataset ready: {srt_path}", "done")
    return srt_path
