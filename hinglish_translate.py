"""
hinglish_translate.py
---------------------
English → Hinglish translation module for WhisperX pipeline.

Pipeline:
  English text
      │
      ▼
  Helsinki-NLP/opus-mt-en-hi   (English → Hindi Devanagari)
      │
      ▼
  indic-transliteration         (Hindi Devanagari → Roman/Hinglish)
      │
      ▼
  Code-mix blender              (keeps English proper nouns / tech terms)
      │
      ▼
  Hinglish text

Install dependencies:
    pip install transformers torch sentencepiece indic-transliteration
"""

import re
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-load heavy models so import is fast
# ---------------------------------------------------------------------------
_translator = None
_transliterator = None

# English words/phrases to preserve as-is (not translate).
# Extend this list for your domain (tech terms, names, etc.)
ENGLISH_PRESERVE = {
    # Common meeting / tech terms
    "meeting", "agenda", "minutes", "action item", "deadline", "project",
    "update", "report", "feedback", "review", "sprint", "deadline",
    "email", "slack", "zoom", "teams", "google", "microsoft",
    "ai", "ml", "nlp", "api", "sdk", "ui", "ux", "app", "data",
    "server", "database", "cloud", "deployment", "pipeline",
    # Common English connectors already used in Hinglish speech
    "ok", "okay", "yes", "no", "please", "thanks", "sorry",
}

# Words that, if found in the Devanagari output, should stay in English
# (identified by matching the source token)
_DIGIT_RE = re.compile(r"\d+")
_ENGLISH_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*")


def _load_models():
    """Load translation and transliteration models (called once, lazily)."""
    global _translator, _transliterator

    if _translator is None:
        try:
            from transformers import pipeline as hf_pipeline
            logger.info("Loading Helsinki-NLP/opus-mt-en-hi …")
            _translator = hf_pipeline(
                "translation",
                model="Helsinki-NLP/opus-mt-en-hi",
                device=-1,          # CPU; change to 0 for GPU
                max_length=512,
            )
            logger.info("Translation model loaded.")
        except Exception as e:
            logger.error(f"Failed to load translation model: {e}")
            raise

    if _transliterator is None:
        try:
            from indic_transliteration import sanscript
            from indic_transliteration.sanscript import transliterate
            _transliterator = (sanscript, transliterate)
            logger.info("Transliterator loaded.")
        except ImportError:
            logger.warning(
                "indic-transliteration not installed. "
                "Devanagari output will be returned as-is. "
                "Install with: pip install indic-transliteration"
            )
            _transliterator = None


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _split_preserve(text: str):
    """
    Split input text into (token, should_preserve) pairs.
    English words in ENGLISH_PRESERVE or ALL_CAPS abbreviations are flagged.
    """
    tokens = text.split()
    result = []
    for tok in tokens:
        clean = re.sub(r"[^A-Za-z0-9]", "", tok).lower()
        preserve = (
            clean in ENGLISH_PRESERVE
            or (tok.isupper() and len(tok) > 1)   # abbreviations like NLP, API
            or bool(_DIGIT_RE.fullmatch(clean))    # numbers
        )
        result.append((tok, preserve))
    return result


def _translate_chunk(text: str) -> str:
    """Translate a plain English string to Hindi Devanagari."""
    if not text.strip():
        return text
    out = _translator(text, max_length=512)
    return out[0]["translation_text"]


def _devanagari_to_roman(text: str) -> str:
    """
    Convert Devanagari script to IAST Roman using indic-transliteration.
    Falls back to returning the original text if library is unavailable.
    """
    if _transliterator is None:
        return text
    sanscript, transliterate = _transliterator
    try:
        roman = transliterate(text, sanscript.DEVANAGARI, sanscript.ITRANS)
        # ITRANS uses uppercase for some letters — normalise to lowercase
        return roman.lower()
    except Exception as e:
        logger.warning(f"Transliteration failed: {e}")
        return text


def _blend_hinglish(source_tokens, translated: str) -> str:
    """
    Blend preserved English tokens back into the translated Hinglish string.

    Strategy:
      - Translated string is split by whitespace.
      - Preserved tokens from source are re-injected at rough proportional
        positions so proper nouns / tech terms stay in English.
    """
    preserved = [tok for tok, keep in source_tokens if keep]
    if not preserved:
        return translated

    trans_tokens = translated.split()
    total_src = len(source_tokens)
    total_tgt = len(trans_tokens)

    for orig_idx, (tok, keep) in enumerate(source_tokens):
        if not keep:
            continue
        # Map source position → target position proportionally
        tgt_pos = int(orig_idx / max(total_src, 1) * max(total_tgt, 1))
        tgt_pos = min(tgt_pos, len(trans_tokens))
        trans_tokens.insert(tgt_pos, tok)

    return " ".join(trans_tokens)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def translate_to_hinglish(text: str) -> str:
    """
    Translate a single English sentence/phrase to Hinglish.

    Args:
        text: English text string.

    Returns:
        Hinglish (code-mixed Roman script) string.

    Example:
        >>> translate_to_hinglish("The meeting is scheduled for tomorrow.")
        "meeting kal ke liye schedule hai"
    """
    _load_models()

    if not text or not text.strip():
        return text

    # 1. Identify tokens to preserve
    source_tokens = _split_preserve(text)
    translatable = " ".join(tok for tok, keep in source_tokens if not keep)

    if not translatable.strip():
        # Entire sentence is preservable English (e.g. "API NLP SDK")
        return text

    # 2. Translate English → Hindi Devanagari
    hindi_devanagari = _translate_chunk(translatable)

    # 3. Devanagari → Roman transliteration
    hindi_roman = _devanagari_to_roman(hindi_devanagari)

    # 4. Blend preserved English tokens back
    hinglish = _blend_hinglish(source_tokens, hindi_roman)

    return hinglish


def translate_segments(
    segments: List[Dict[str, Any]],
    text_key: str = "text",
    output_key: str = "hinglish_text",
) -> List[Dict[str, Any]]:
    """
    Translate all segments from a WhisperX result dict in-place.

    Args:
        segments:   List of segment dicts from whisperx model.transcribe().
        text_key:   Key in each segment containing English text.
        output_key: Key to write Hinglish translation into.

    Returns:
        Same segments list with `output_key` added to each segment.

    Example:
        >>> result = model.transcribe(audio, batch_size=16)
        >>> result["segments"] = translate_segments(result["segments"])
    """
    _load_models()
    total = len(segments)
    for i, seg in enumerate(segments):
        src = seg.get(text_key, "").strip()
        if src:
            try:
                seg[output_key] = translate_to_hinglish(src)
            except Exception as e:
                logger.warning(f"Segment {i}/{total} translation failed: {e}")
                seg[output_key] = src   # fallback: keep English
        else:
            seg[output_key] = ""

        if (i + 1) % 10 == 0 or (i + 1) == total:
            logger.info(f"Translated {i+1}/{total} segments.")

    return segments


def translate_live_segment(segment: Dict[str, Any]) -> Dict[str, Any]:
    """
    Translate a single segment dict. Designed for real-time/streaming use.

    Args:
        segment: Single WhisperX segment dict with at least a "text" key.

    Returns:
        Segment dict with "hinglish_text" added.

    Example (in a live loop):
        for seg in live_stream_segments():
            seg = translate_live_segment(seg)
            display_caption(seg["hinglish_text"], seg["start"], seg["end"])
    """
    _load_models()
    src = segment.get("text", "").strip()
    segment["hinglish_text"] = translate_to_hinglish(src) if src else ""
    return segment


# ---------------------------------------------------------------------------
# Quick test / demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    test_sentences = [
        "The meeting is scheduled for tomorrow morning.",
        "Please send the report by end of day.",
        "The NLP pipeline is working correctly.",
        "We need to update the API documentation.",
        "Can you review the sprint backlog before the standup?",
        "The deadline for this project is next Friday.",
    ]

    print("\n=== English → Hinglish Translation Demo ===\n")
    for sent in test_sentences:
        hinglish = translate_to_hinglish(sent)
        print(f"  EN : {sent}")
        print(f"  HI : {hinglish}")
        print()
