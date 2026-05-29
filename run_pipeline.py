"""
run_pipeline.py
---------------
Full pipeline: WhisperX  →  Hinglish Translation  →  Live Captions  →  Summary

Usage:
    # Transcribe a recorded meeting file
    python run_pipeline.py --audio meeting.wav

    # Live microphone input (requires pyaudio)
    python run_pipeline.py --live

    # Choose summarization backend
    python run_pipeline.py --audio meeting.wav --backend anthropic

Install all dependencies:
    pip install whisperx transformers torch sentencepiece indic-transliteration
    pip install anthropic          # optional, for best summary quality
    pip install openai             # optional, alternative API backend
    pip install pyaudio            # optional, for live mic input
"""

import argparse
import logging
import sys
from pathlib import Path

from hinglish_translate import translate_segments, translate_live_segment
from summarizer import MeetingSummarizer, format_summary

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("pipeline")


# ---------------------------------------------------------------------------
# File-based pipeline (recorded meeting)
# ---------------------------------------------------------------------------

def run_file_pipeline(
    audio_path: str,
    device: str = "cpu",
    batch_size: int = 8,
    compute_type: str = "int8",
    hf_token: str = None,
    backend: str = "auto",
    window_minutes: float = 5.0,
):
    import whisperx

    logger.info(f"Loading WhisperX model …")
    model = whisperx.load_model(
        "large-v2",
        device=device,
        compute_type=compute_type,
    )

    logger.info(f"Loading audio: {audio_path}")
    audio = whisperx.load_audio(audio_path)

    # ── Step 1: Transcribe ────────────────────────────────────────────────
    logger.info("Transcribing …")
    result = model.transcribe(audio, batch_size=batch_size, language="en")
    logger.info(f"Got {len(result['segments'])} segments.")

    # ── Step 2: Align (word-level timestamps) ────────────────────────────
    logger.info("Aligning …")
    model_a, metadata = whisperx.load_align_model(
        language_code="en", device=device
    )
    result = whisperx.align(
        result["segments"], model_a, metadata, audio, device,
        return_char_alignments=False,
    )

    # ── Step 3: Diarize (who spoke when) ─────────────────────────────────
    if hf_token:
        logger.info("Diarizing …")
        from whisperx.diarize import DiarizationPipeline
        diarize_model = DiarizationPipeline(token=hf_token, device=device)
        diarize_segments = diarize_model(audio)
        result = whisperx.assign_word_speakers(diarize_segments, result)
    else:
        logger.warning("No HF token provided — skipping diarization.")

    # ── Step 4: Translate to Hinglish ─────────────────────────────────────
    logger.info("Translating to Hinglish …")
    segments = translate_segments(result["segments"])

    # Print all captions
    print("\n" + "═" * 60)
    print("HINGLISH CAPTIONS")
    print("═" * 60)
    for seg in segments:
        start = seg.get("start", 0)
        end   = seg.get("end", 0)
        spk   = seg.get("speaker", "")
        label = f"[{spk}] " if spk else ""
        print(f"  {_fmt(start)} → {_fmt(end)}  {label}{seg['hinglish_text']}")
    print("═" * 60 + "\n")

    # ── Step 5: Summarize ─────────────────────────────────────────────────
    logger.info("Summarizing …")
    summarizer = MeetingSummarizer(backend=backend, window_minutes=window_minutes)
    for seg in segments:
        chunk_summary = summarizer.add_segment(seg)
        if chunk_summary:
            print("\n📌  Periodic Summary:")
            print(format_summary(chunk_summary))
            print()

    # Final full-meeting summary
    final = summarizer.full_meeting_summary()
    if final:
        print("\n" + "═" * 60)
        print("FINAL MEETING SUMMARY")
        print("═" * 60)
        print(format_summary(final))
        print("═" * 60 + "\n")

    return segments, final


# ---------------------------------------------------------------------------
# Live microphone pipeline (streaming)
# ---------------------------------------------------------------------------

def run_live_pipeline(
    device: str = "cpu",
    backend: str = "auto",
    window_minutes: float = 5.0,
    sample_rate: int = 16000,
    chunk_seconds: int = 5,
):
    """
    Capture mic audio in chunks, transcribe each chunk, translate, display.
    Requires: pyaudio  (pip install pyaudio)
    """
    try:
        import pyaudio
        import numpy as np
    except ImportError:
        logger.error("pyaudio / numpy not installed. Run: pip install pyaudio numpy")
        sys.exit(1)

    import whisperx

    logger.info("Loading WhisperX for live transcription …")
    model = whisperx.load_model("base", device=device, compute_type="int8")
    summarizer = MeetingSummarizer(backend=backend, window_minutes=window_minutes)

    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=sample_rate,
        input=True,
        frames_per_buffer=1024,
    )

    print("\n🎙  Live captioning started. Press Ctrl+C to stop.\n")
    elapsed = 0.0
    try:
        while True:
            # Collect audio chunk
            frames = []
            for _ in range(int(sample_rate / 1024 * chunk_seconds)):
                frames.append(stream.read(1024, exception_on_overflow=False))

            raw = b"".join(frames)
            audio_np = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

            # Transcribe chunk
            result = model.transcribe(audio_np, batch_size=1, language="en")
            segs = result.get("segments", [])

            # Offset timestamps
            for seg in segs:
                seg["start"] = seg.get("start", 0) + elapsed
                seg["end"]   = seg.get("end", 0)   + elapsed

            elapsed += chunk_seconds

            for seg in segs:
                # Translate
                seg = translate_live_segment(seg)
                # Display
                start, end = seg.get("start", 0), seg.get("end", 0)
                print(f"  [{_fmt(start)} → {_fmt(end)}]  {seg['hinglish_text']}")
                # Feed summarizer
                chunk_sum = summarizer.add_segment(seg)
                if chunk_sum:
                    print("\n📌  Summary so far:")
                    print(format_summary(chunk_sum))
                    print()

    except KeyboardInterrupt:
        print("\n\n🛑  Stopped. Generating final summary …\n")
        final = summarizer.flush()
        if final:
            print("═" * 60)
            print(format_summary(final))
            print("═" * 60)
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _fmt(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WhisperX + Hinglish + Summary pipeline")
    parser.add_argument("--audio",    type=str,  help="Path to audio file (WAV/MP3/etc.)")
    parser.add_argument("--live",     action="store_true", help="Use live microphone input")
    parser.add_argument("--device",   type=str,  default="cpu",  choices=["cpu", "cuda"])
    parser.add_argument("--batch",    type=int,  default=8,      help="Batch size for transcription")
    parser.add_argument("--backend",  type=str,  default="auto",
                        choices=["auto", "anthropic", "openai", "bart", "mt5"],
                        help="Summarization backend")
    parser.add_argument("--window",   type=float, default=5.0,
                        help="Summary window in minutes (default 5)")
    parser.add_argument("--hf_token", type=str,  default=None,
                        help="HuggingFace token for speaker diarization")
    args = parser.parse_args()

    if args.live:
        run_live_pipeline(
            device=args.device,
            backend=args.backend,
            window_minutes=args.window,
        )
    elif args.audio:
        if not Path(args.audio).exists():
            logger.error(f"Audio file not found: {args.audio}")
            sys.exit(1)
        run_file_pipeline(
            audio_path=args.audio,
            device=args.device,
            batch_size=args.batch,
            backend=args.backend,
            window_minutes=args.window,
            hf_token=args.hf_token,
        )
    else:
        parser.print_help()
