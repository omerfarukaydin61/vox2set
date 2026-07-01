"""Subtitle translation that survives word-order differences between languages.

Translating each .srt cue independently is wrong: a cue is a *fragment* of a
sentence, so the model loses context, and because languages reorder words
(e.g. Turkish puts the verb last) the translation of cue N may belong at the
end of the sentence — you can't drop it back into cue N's time slot.

So we never map source cue -> target cue 1:1. Instead:
  1. merge consecutive cues into whole **sentences** (keeping each sentence's
     [start, end] time window),
  2. translate the sentence *with context* (NLLB-200, a local offline model),
  3. **reflow** the translated sentence into fresh cues sized for readability
     and timed proportionally *inside that sentence's window*.
A sentence keeps the same time window it had in the source, so the result stays
in sync with the video no matter how the words reorder inside it.

Heavy deps (torch/transformers) are imported lazily so the module (and its
FLORES map, used by the API for validation) stays import-light.
"""

# ISO 639-1 -> NLLB FLORES-200 code. The UI offers these as translation targets
# and the API validates against this map. Extend as needed.
FLORES = {
    "en": "eng_Latn", "tr": "tur_Latn", "de": "deu_Latn", "fr": "fra_Latn",
    "es": "spa_Latn", "it": "ita_Latn", "pt": "por_Latn", "nl": "nld_Latn",
    "ru": "rus_Cyrl", "uk": "ukr_Cyrl", "pl": "pol_Latn", "cs": "ces_Latn",
    "ro": "ron_Latn", "el": "ell_Grek", "sv": "swe_Latn", "da": "dan_Latn",
    "fi": "fin_Latn", "hu": "hun_Latn", "ar": "arb_Arab", "fa": "pes_Arab",
    "he": "heb_Hebr", "hi": "hin_Deva", "ja": "jpn_Jpan", "ko": "kor_Hang",
    "zh": "zho_Hans", "id": "ind_Latn", "vi": "vie_Latn", "th": "tha_Thai",
    "az": "azj_Latn",
}

# Subtitle shaping defaults (cosmetic / readability).
MAX_LINE = 42        # characters per line
MAX_LINES = 2        # lines per cue
_ENDERS = ".!?…"
_TRAILERS = "\"'”’)]》」』"


def _is_sentence_end(text):
    t = text.rstrip()
    while t and t[-1] in _TRAILERS:
        t = t[:-1].rstrip()
    return bool(t) and t[-1] in _ENDERS


def merge_sentences(segments):
    """Group consecutive cue dicts ({start,end,text}) into sentences, each with
    its own [start, end] window. A sentence runs until a cue's text ends with
    terminal punctuation (cue boundaries are Whisper's natural phrase breaks)."""
    sentences, buf, start, end, spk = [], [], None, None, None
    for s in segments:
        txt = (s.get("text") or "").strip()
        if not txt:
            continue
        if start is None:
            start = s["start"]
        if spk is None:                       # the sentence's speaker (first labeled cue)
            spk = s.get("speaker")
        buf.append(txt)
        end = s["end"]
        if _is_sentence_end(txt):
            sentences.append({"text": " ".join(buf), "start": start, "end": end, "speaker": spk})
            buf, start, spk = [], None, None
    if buf:
        sentences.append({"text": " ".join(buf), "start": start, "end": end, "speaker": spk})
    return sentences


def _wrap(text, width):
    """Greedily pack words into chunks no wider than `width` (on word bounds)."""
    out, cur = [], ""
    for w in text.split():
        if cur and len(cur) + 1 + len(w) > width:
            out.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        out.append(cur)
    return out or [text.strip()]


def reflow(text, start, end, max_line=MAX_LINE, max_lines=MAX_LINES):
    """Split a translated sentence into cues and spread [start, end] across them
    proportional to character length (longer chunk -> more time)."""
    chunks = _wrap(text, max_line * max_lines)
    total = sum(len(c) for c in chunks) or 1
    span = max(0.001, end - start)
    cues, t, acc = [], start, 0
    for i, c in enumerate(chunks):
        acc += len(c)
        e = end if i == len(chunks) - 1 else round(start + span * acc / total, 3)
        if e <= t:
            e = min(end, round(t + 0.1, 3))
        lines = _wrap(c, max_line)[:max_lines]
        cues.append({"start": t, "end": e, "text": "\n".join(lines)})
        t = e
    return cues


# --------------------------------------------------------------------------- #
# NLLB (lazy heavy imports)
# --------------------------------------------------------------------------- #
def _load_model(model_name):
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    model.eval()
    return tok, model, device


def _translate_texts(texts, src_flores, tgt_flores, tok, model, device,
                     batch_size=8, on_frac=None):
    import torch
    tok.src_lang = src_flores
    bos = tok.convert_tokens_to_ids(tgt_flores)
    out, n = [], len(texts)
    for i in range(0, n, batch_size):
        batch = texts[i:i + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                  max_length=512).to(device)
        with torch.no_grad():
            gen = model.generate(**enc, forced_bos_token_id=bos,
                                 max_length=512, num_beams=2)
        out += tok.batch_decode(gen, skip_special_tokens=True)
        if on_frac:
            on_frac(min(1.0, (i + len(batch)) / n))
    return out


def translate_srt(segments, src_iso, tgt_iso, *, model_name, progress=None,
                  max_line=MAX_LINE, max_lines=MAX_LINES, batch_size=8):
    """segments (from parse_srt) -> translated, reflowed cues for write_srt.

    `progress(stage, frac, msg)` is an optional callback. Raises ValueError for
    an unsupported source/target language.
    """
    src_f = FLORES.get((src_iso or "").lower())
    tgt_f = FLORES.get((tgt_iso or "").lower())
    if not src_f:
        raise ValueError(f"Unsupported source language: {src_iso}")
    if not tgt_f:
        raise ValueError(f"Unsupported target language: {tgt_iso}")

    sentences = merge_sentences(segments)
    if not sentences:
        return []

    if progress:
        progress("load", 0.05, "Loading translation model…")
    tok, model, device = _load_model(model_name)

    if progress:
        progress("translate", 0.10, f"Translating {len(sentences)} sentences…")
    texts = [s["text"] for s in sentences]
    translations = _translate_texts(
        texts, src_f, tgt_f, tok, model, device, batch_size,
        on_frac=(lambda f: progress("translate", 0.10 + 0.85 * f, None)) if progress else None)

    cues = []
    for s, tr in zip(sentences, translations):
        chunk = reflow(tr, s["start"], s["end"], max_line, max_lines)
        for c in chunk:                       # carry the speaker onto every reflowed cue
            c["speaker"] = s.get("speaker")
        cues.extend(chunk)
    if progress:
        progress("write", 0.97, "Writing subtitles…")
    return cues
