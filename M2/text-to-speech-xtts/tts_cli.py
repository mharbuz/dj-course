#!/usr/bin/env python3
"""
CLI for single TTS call. Used by azor-chatdog-js /audio command.
Reads text from stdin when --text - is passed.
"""

import argparse
import sys
from pathlib import Path

# Suppress Coqui/Torch warnings when running headless
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    default_speaker = script_dir / "sample-agent.wav"

    parser = argparse.ArgumentParser(description="Synthesize text to WAV using XTTS")
    parser.add_argument(
        "--text",
        required=True,
        help="Text to synthesize. Use '-' to read from stdin (recommended for long/complex text).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output WAV file path.",
    )
    parser.add_argument(
        "--speaker-wav",
        default=str(default_speaker),
        help=f"Reference speaker WAV path (default: {default_speaker}).",
    )
    parser.add_argument(
        "--language",
        default="pl",
        help="Language code (default: pl).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Minimal stdout (only errors).",
    )
    args = parser.parse_args()

    text = args.text
    if text.strip() == "-":
        text = sys.stdin.read()

    text = text.strip()
    if not text:
        if not args.quiet:
            print("Error: empty text", file=sys.stderr)
        return 1

    try:
        if not args.quiet:
            print("Loading XTTS model...", file=sys.stderr)
        from TTS.api import TTS

        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=not args.quiet).to("cpu")
        if not args.quiet:
            print("Generating audio...", file=sys.stderr)

        tts.tts_to_file(
            text=text,
            file_path=args.output,
            speaker_wav=args.speaker_wav,
            language=args.language,
        )
        if not args.quiet:
            print(f"Saved: {args.output}", file=sys.stderr)
        return 0
    except Exception as e:
        print(f"TTS error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
