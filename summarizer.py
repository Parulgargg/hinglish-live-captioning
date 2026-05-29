"""
summarizer.py
-------------
Meeting summarization module for WhisperX + Hinglish pipeline.

Supports three modes:
  1. Periodic (time-window)  — summarize every N minutes of transcript
  2. On-demand               — summarize the full session at end of meeting
  3. Streaming               — accumulate segments, flush when buffer is full

Summary output:
  - 3–5 bullet points in Hinglish
  - Key decisions extracted
  - Action items with owner names (from diarization speaker labels)
  - Overall topic/title

Model options (in order of quality):
  A. Claude / OpenAI API (best quality, needs API key)
  B. facebook/bart-large-cnn   (offline, English only — then translate output)
  C. google/mt5-small           (offline, multilingual, lower quality)

Install:
    pip install transformers torch sentencepiece anthropic openai
"""

import os
import re
import time
import logging
from typing import List, Dict, Any, Optional, Literal
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MeetingSegment:
    start: float          # seconds
    end: float            # seconds
    speaker: str          # e.g. "SPEAKER_00"
    text: str             # original English
    hinglish_text: str    # translated Hinglish


@dataclass
class MeetingSummary:
    window_start: float
    window_end: float
    title: str
    bullets: List[str]
    decisions: List[str]
    action_items: List[str]       # "Owner: action"
    raw_hinglish_summary: str     # full paragraph summary in Hinglish


# ---------------------------------------------------------------------------
# Backend selector
# ---------------------------------------------------------------------------

SummaryBackend = Literal["anthropic", "openai", "bart", "mt5", "auto"]

_bart_pipeline = None
_mt5_pipeline  = None


def _load_bart():
    global _bart_pipeline
    if _bart_pipeline is None:
        from transformers import pipeline as hf_pipeline
        logger.info("Loading facebook/bart-large-cnn …")
        _bart_pipeline = hf_pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            device=-1,
            max_length=256,
            min_length=60,
            do_sample=False,
        )
        logger.info("BART loaded.")
    return _bart_pipeline


def _load_mt5():
    global _mt5_pipeline
    if _mt5_pipeline is None:
        from transformers import pipeline as hf_pipeline
        logger.info("Loading csebuetnlp/mT5_multilingual_XLSum …")
        _mt5_pipeline = hf_pipeline(
            "summarization",
            model="csebuetnlp/mT5_multilingual_XLSum",
            device=-1,
            max_length=256,
            min_length=40,
        )
        logger.info("mT5 loaded.")
    return _mt5_pipeline


def _detect_backend() -> SummaryBackend:
    """Auto-detect best available backend."""
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    return "bart"   # offline fallback


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are a meeting assistant. You will receive a Hinglish meeting transcript 
(code-mixed Hindi + English as spoken in Indian workplaces).

Your job is to summarise the meeting. Reply ONLY with valid JSON in this exact format:
{
  "title": "<short meeting topic, 5-8 words>",
  "bullets": ["<point 1>", "<point 2>", "<point 3>"],
  "decisions": ["<decision 1>", ...],
  "action_items": ["<SPEAKER_NAME>: <action>", ...],
  "hinglish_summary": "<2-3 sentence summary in Hinglish>"
}

Rules:
- bullets: 3–5 key discussion points in Hinglish
- decisions: list firm decisions made (empty list if none)
- action_items: list tasks assigned to specific people; use speaker label if no name known
- hinglish_summary: natural Hinglish paragraph (Roman script)
- Do NOT add any text outside the JSON block
"""


def _build_user_prompt(segments: List[MeetingSegment]) -> str:
    lines = []
    for seg in segments:
        ts = f"[{_fmt_time(seg.start)} → {_fmt_time(seg.end)}]"
        text = seg.hinglish_text if seg.hinglish_text else seg.text
        lines.append(f"{seg.speaker} {ts}: {text}")
    transcript = "\n".join(lines)
    return f"Here is the meeting transcript:\n\n{transcript}\n\nPlease summarise."


def _fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


# ---------------------------------------------------------------------------
# API backends
# ---------------------------------------------------------------------------

def _summarise_anthropic(prompt: str) -> Dict[str, Any]:
    import anthropic
    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.content[0].text
    return _parse_json_response(raw)


def _summarise_openai(prompt: str) -> Dict[str, Any]:
    from openai import OpenAI
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        max_tokens=1024,
        temperature=0.3,
    )
    raw = response.choices[0].message.content
    return _parse_json_response(raw)


def _summarise_bart(segments: List[MeetingSegment]) -> Dict[str, Any]:
    """Offline BART — English only, then manually structure output."""
    pipe = _load_bart()
    # Concatenate English text (BART works on English)
    full_text = " ".join(
        seg.text for seg in segments if seg.text.strip()
    )
    if not full_text.strip():
        return _empty_summary_dict()

    # BART max input ~1024 tokens — truncate if needed
    words = full_text.split()
    if len(words) > 900:
        full_text = " ".join(words[:900])

    result = pipe(full_text)[0]["summary_text"]

    # Structure the flat summary into our format
    sentences = [s.strip() for s in re.split(r"[.!?]", result) if s.strip()]
    return {
        "title": sentences[0][:60] if sentences else "Meeting Summary",
        "bullets": sentences[:5],
        "decisions": [],
        "action_items": [],
        "hinglish_summary": result,
    }


def _summarise_mt5(segments: List[MeetingSegment]) -> Dict[str, Any]:
    """Offline mT5 multilingual — handles mixed text better than BART."""
    pipe = _load_mt5()
    full_text = " ".join(
        (seg.hinglish_text or seg.text) for seg in segments if seg.text.strip()
    )
    if not full_text.strip():
        return _empty_summary_dict()

    words = full_text.split()
    if len(words) > 700:
        full_text = " ".join(words[:700])

    result = pipe(full_text)[0]["summary_text"]
    sentences = [s.strip() for s in re.split(r"[.!?।]", result) if s.strip()]
    return {
        "title": sentences[0][:60] if sentences else "Meeting Summary",
        "bullets": sentences[:5],
        "decisions": [],
        "action_items": [],
        "hinglish_summary": result,
    }


# ---------------------------------------------------------------------------
# JSON parsing helper
# ---------------------------------------------------------------------------

def _parse_json_response(raw: str) -> Dict[str, Any]:
    import json
    # Strip markdown code fences if present
    cleaned = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parse failed ({e}), returning raw text.")
        return {
            "title": "Meeting Summary",
            "bullets": [cleaned[:200]],
            "decisions": [],
            "action_items": [],
            "hinglish_summary": cleaned,
        }


def _empty_summary_dict() -> Dict[str, Any]:
    return {
        "title": "No content",
        "bullets": [],
        "decisions": [],
        "action_items": [],
        "hinglish_summary": "",
    }


# ---------------------------------------------------------------------------
# Main summarizer class
# ---------------------------------------------------------------------------

class MeetingSummarizer:
    """
    Accumulates WhisperX segments and generates structured meeting summaries.

    Usage — periodic (every 5 minutes of audio):
        summarizer = MeetingSummarizer(window_minutes=5)
        for seg in whisperx_segments:
            summary = summarizer.add_segment(seg)
            if summary:
                print(summary)   # triggered every 5 min

    Usage — on-demand (end of meeting):
        summarizer = MeetingSummarizer()
        for seg in whisperx_segments:
            summarizer.add_segment(seg)
        final = summarizer.flush()
        print(final)
    """

    def __init__(
        self,
        backend: SummaryBackend = "auto",
        window_minutes: float = 5.0,
        min_words_to_summarise: int = 50,
    ):
        """
        Args:
            backend:               "auto" | "anthropic" | "openai" | "bart" | "mt5"
            window_minutes:        How many minutes of audio per summary chunk.
                                   Set to 0 or None to disable periodic summaries.
            min_words_to_summarise: Don't summarise if transcript has fewer words.
        """
        self.backend = _detect_backend() if backend == "auto" else backend
        self.window_seconds = (window_minutes or 0) * 60
        self.min_words = min_words_to_summarise

        self._buffer: List[MeetingSegment] = []
        self._window_start: Optional[float] = None
        self._all_segments: List[MeetingSegment] = []

        logger.info(f"MeetingSummarizer ready | backend={self.backend} | window={window_minutes}min")

    # ------------------------------------------------------------------
    def add_segment(self, seg: Dict[str, Any]) -> Optional[MeetingSummary]:
        """
        Add a single WhisperX segment (dict) to the buffer.
        Returns a MeetingSummary if the time window is complete, else None.

        Expected keys in seg dict:
            start, end, text, hinglish_text (added by hinglish_translate.py),
            speaker (added by diarize.py — optional)
        """
        ms = MeetingSegment(
            start=seg.get("start", 0.0),
            end=seg.get("end", 0.0),
            speaker=seg.get("speaker", "SPEAKER_00"),
            text=seg.get("text", ""),
            hinglish_text=seg.get("hinglish_text", seg.get("text", "")),
        )
        self._buffer.append(ms)
        self._all_segments.append(ms)

        if self._window_start is None:
            self._window_start = ms.start

        # Check if window is full
        if (
            self.window_seconds > 0
            and (ms.end - self._window_start) >= self.window_seconds
        ):
            summary = self._summarise_buffer()
            self._buffer = []
            self._window_start = None
            return summary

        return None

    # ------------------------------------------------------------------
    def flush(self) -> Optional[MeetingSummary]:
        """
        Force-summarise whatever is in the buffer (call at end of meeting).
        Returns None if buffer is empty or too short.
        """
        if not self._buffer:
            return None
        summary = self._summarise_buffer()
        self._buffer = []
        self._window_start = None
        return summary

    # ------------------------------------------------------------------
    def full_meeting_summary(self) -> Optional[MeetingSummary]:
        """
        Summarise the entire meeting (all segments seen so far).
        Call once after the meeting ends for a complete summary.
        """
        if not self._all_segments:
            return None
        return self._run_summariser(self._all_segments)

    # ------------------------------------------------------------------
    def _summarise_buffer(self) -> Optional[MeetingSummary]:
        word_count = sum(len(s.text.split()) for s in self._buffer)
        if word_count < self.min_words:
            logger.info(f"Buffer too short ({word_count} words), skipping summary.")
            return None
        return self._run_summariser(self._buffer)

    # ------------------------------------------------------------------
    def _run_summariser(self, segments: List[MeetingSegment]) -> MeetingSummary:
        w_start = segments[0].start
        w_end   = segments[-1].end

        try:
            if self.backend in ("anthropic", "openai"):
                prompt = _build_user_prompt(segments)
                if self.backend == "anthropic":
                    data = _summarise_anthropic(prompt)
                else:
                    data = _summarise_openai(prompt)
            elif self.backend == "mt5":
                data = _summarise_mt5(segments)
            else:
                data = _summarise_bart(segments)

        except Exception as e:
            logger.error(f"Summarisation failed: {e}")
            data = _empty_summary_dict()

        return MeetingSummary(
            window_start=w_start,
            window_end=w_end,
            title=data.get("title", "Meeting Summary"),
            bullets=data.get("bullets", []),
            decisions=data.get("decisions", []),
            action_items=data.get("action_items", []),
            raw_hinglish_summary=data.get("hinglish_summary", ""),
        )


# ---------------------------------------------------------------------------
# Convenience function (no class needed for simple use)
# ---------------------------------------------------------------------------

def summarize_segments(
    segments: List[Dict[str, Any]],
    backend: SummaryBackend = "auto",
) -> MeetingSummary:
    """
    One-shot summarise a list of WhisperX segments (after Hinglish translation).

    Args:
        segments: List of segment dicts (must have "hinglish_text" key).
        backend:  Which summarisation backend to use.

    Returns:
        MeetingSummary dataclass.

    Example:
        >>> result["segments"] = translate_segments(result["segments"])
        >>> summary = summarize_segments(result["segments"])
        >>> print(summary.title)
        >>> for b in summary.bullets: print("•", b)
    """
    summarizer = MeetingSummarizer(backend=backend, window_minutes=0)
    for seg in segments:
        summarizer.add_segment(seg)
    return summarizer.flush()


def format_summary(summary: MeetingSummary) -> str:
    """Pretty-print a MeetingSummary to a readable string."""
    lines = [
        f"📋  {summary.title}",
        f"    {_fmt_time(summary.window_start)} → {_fmt_time(summary.window_end)}",
        "",
    ]
    if summary.bullets:
        lines.append("Key Points:")
        for b in summary.bullets:
            lines.append(f"  • {b}")
        lines.append("")
    if summary.decisions:
        lines.append("Decisions:")
        for d in summary.decisions:
            lines.append(f"  ✓ {d}")
        lines.append("")
    if summary.action_items:
        lines.append("Action Items:")
        for a in summary.action_items:
            lines.append(f"  → {a}")
        lines.append("")
    if summary.raw_hinglish_summary:
        lines.append("Summary:")
        lines.append(f"  {summary.raw_hinglish_summary}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Quick test / demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Simulate WhisperX segments already translated to Hinglish
    fake_segments = [
        {"start": 0.0,   "end": 8.0,  "speaker": "Rahul",  "text": "Let's discuss the project deadline.",        "hinglish_text": "project ka deadline discuss karte hain."},
        {"start": 8.5,   "end": 18.0, "speaker": "Priya",  "text": "The API integration is almost done.",        "hinglish_text": "API integration almost ho gayi hai."},
        {"start": 18.5,  "end": 30.0, "speaker": "Rahul",  "text": "We need to review the sprint backlog.",      "hinglish_text": "sprint backlog review karna padega."},
        {"start": 31.0,  "end": 44.0, "speaker": "Ankit",  "text": "I will send the updated report by Friday.",  "hinglish_text": "main Friday tak updated report bhejunga."},
        {"start": 45.0,  "end": 58.0, "speaker": "Priya",  "text": "The NLP model accuracy is at 87 percent.",  "hinglish_text": "NLP model ki accuracy 87 percent hai."},
        {"start": 59.0,  "end": 72.0, "speaker": "Rahul",  "text": "Okay, let's schedule the next meeting.",    "hinglish_text": "okay, next meeting schedule karte hain."},
    ]

    print("\n=== Meeting Summarizer Demo (offline BART) ===\n")
    try:
        summary = summarize_segments(fake_segments, backend="bart")
        print(format_summary(summary))
    except Exception as e:
        print(f"(Demo requires 'transformers' installed: {e})")
        # Show what the output structure looks like anyway
        dummy = MeetingSummary(
            window_start=0.0,
            window_end=72.0,
            title="Sprint review aur deadline discussion",
            bullets=[
                "API integration almost complete hai",
                "Sprint backlog review karna hai",
                "NLP model 87% accurate hai",
            ],
            decisions=["Next meeting schedule hogi"],
            action_items=["Ankit: Friday tak report bhejna hai"],
            raw_hinglish_summary=(
                "Team ne project deadline aur API integration discuss ki. "
                "Ankit Friday tak report bhejega. "
                "Next meeting jaldi schedule hogi."
            ),
        )
        print(format_summary(dummy))
