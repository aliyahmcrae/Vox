#!/usr/bin/env python3
"""
Generate short spoken MP3 responses for each intent.

Usage:
  python3 voice.py
  python3 voice.py --force

Requirements:
  export OPENAI_API_KEY=...

  pip install openai kokoro soundfile numpy pydub

  ffmpeg must also be installed:
    Ubuntu: sudo apt install ffmpeg
    macOS:  brew install ffmpeg

Optional:
  sudo apt install espeak-ng

The script:
- loads unique intent names from vox-tiny/intent_names.txt
- asks OpenAI for a short spoken reply for each intent
- synthesizes speech locally using Kokoro
- saves intents/{intent}.mp3
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from openai import OpenAI
from kokoro import KPipeline
from pydub import AudioSegment


DEFAULT_INTENT_FILE = Path("vox-tiny/intent_names.txt")
DEFAULT_OUT_DIR = Path("intents")

# Pick whatever model you're actually using.
# gpt-realtime-mini generally isn't intended for Responses API text generation.
DEFAULT_MODEL = "gpt-5-mini"

KOKORO_LANG = "a"      # American English
KOKORO_VOICE = "af_heart"
KOKORO_SPEED = 1.0
KOKORO_SAMPLE_RATE = 24000


EXAMPLES_TEXT = """Examples:
Listening...
'help me buy a train ticket to boston'
→ transport_ticket (0.755)
Would've played: "I'll check ticket options."

'help me create a new list'
→ lists_createoradd (0.825)
Would've played: "Added it to your list."

'turn off the lights please'
→ iot_hue_lightoff (0.859)
Would've played: "Turning the lights off."
"""


SYSTEM_INSTRUCTION = (
    "You are a concise assistant that writes short, polite, natural spoken "
    "replies for a voice assistant. "
    "For a given intent name produce a short spoken response that a generic "
    "assistant could say out loud. "
    "Keep replies short (about 2-10 words), natural, and unambiguous. "
    "Output only the reply text, without surrounding quotes, metadata, "
    "or explanations."
)


# ---------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------

def create_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY not set in environment"
        )

    return OpenAI(api_key=api_key)


def ask_gpt_for_reply(
    client: OpenAI,
    intent: str,
    model: str,
) -> str:
    """
    Generate a short spoken response for an intent.
    """

    prompt = (
        f"Intent: {intent}\n\n"
        f"Provide a short spoken reply appropriate for this intent."
    )

    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": SYSTEM_INSTRUCTION,
            },
            {
                "role": "user",
                "content": EXAMPLES_TEXT,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        max_output_tokens=40,
    )

    text = response.output_text.strip()

    if (
        text.startswith('"')
        and text.endswith('"')
    ) or (
        text.startswith("'")
        and text.endswith("'")
    ):
        text = text[1:-1].strip()

    text = " ".join(text.split())

    return text


# ---------------------------------------------------------------------
# Intents
# ---------------------------------------------------------------------

def load_intents(intent_file: Path) -> list[str]:
    if not intent_file.exists():
        raise FileNotFoundError(
            f"Intent file not found: {intent_file}"
        )

    with intent_file.open("r", encoding="utf-8") as f:
        lines = [
            line.strip()
            for line in f
            if line.strip()
        ]

    seen = set()
    unique = []

    for intent in lines:
        if intent not in seen:
            seen.add(intent)
            unique.append(intent)

    return unique


# ---------------------------------------------------------------------
# Kokoro
# ---------------------------------------------------------------------

def create_kokoro_pipeline() -> KPipeline:
    """
    Load Kokoro once and reuse it for all intents.
    """
    return KPipeline(lang_code=KOKORO_LANG)


def kokoro_to_mp3(
    text: str,
    out_path: Path,
    pipeline: KPipeline,
    voice: str = KOKORO_VOICE,
):
    """
    Generate speech using Kokoro and save as MP3.
    """

    if not text.strip():
        raise ValueError("Cannot synthesize empty text")

    chunks: list[np.ndarray] = []

    generator = pipeline(
        text,
        voice=voice,
        speed=KOKORO_SPEED,
    )

    for _gs, _ps, audio in generator:
        if hasattr(audio, "numpy"):
            audio = audio.numpy()

        chunks.append(
            np.asarray(audio, dtype=np.float32)
        )

    if not chunks:
        raise RuntimeError(
            "Kokoro produced no audio"
        )

    audio = np.concatenate(chunks)

    wav_path = out_path.with_suffix(".wav")

    sf.write(
        str(wav_path),
        audio,
        KOKORO_SAMPLE_RATE,
    )

    AudioSegment.from_wav(
        str(wav_path)
    ).export(
        str(out_path),
        format="mp3",
        bitrate="128k",
    )

    wav_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate MP3 responses for intents."
    )

    parser.add_argument(
        "--intents-file",
        default=str(DEFAULT_INTENT_FILE),
    )

    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT_DIR),
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
    )

    parser.add_argument(
        "--voice",
        default=KOKORO_VOICE,
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate existing files",
    )

    parser.add_argument(
        "--sleep",
        type=float,
        default=0.5,
        help="Seconds between OpenAI requests",
    )

    args = parser.parse_args()

    try:
        client = create_client()
    except Exception as e:
        print(f"Error creating OpenAI client: {e}")
        sys.exit(1)

    try:
        pipeline = create_kokoro_pipeline()
    except Exception as e:
        print(f"Error loading Kokoro: {e}")
        sys.exit(1)

    intents_file = Path(args.intents_file)
    out_dir = Path(args.out_dir)

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        intents = load_intents(intents_file)
    except Exception as e:
        print(f"Error loading intents: {e}")
        sys.exit(1)

    print(
        f"Loaded {len(intents)} unique intents."
    )
    print(
        f"Writing MP3s to: {out_dir}"
    )

    total = len(intents)

    for idx, intent in enumerate(intents, start=1):

        safe_name = (
            intent
            .replace("/", "_")
            .replace("\\", "_")
            .replace(" ", "_")
        )

        out_path = out_dir / f"{safe_name}.mp3"

        if out_path.exists() and not args.force:
            print(
                f"[{idx}/{total}] "
                f"Skipping existing: {out_path}"
            )
            continue

        try:
            print(
                f"[{idx}/{total}] "
                f"Generating text for: {intent}"
            )

            reply = ask_gpt_for_reply(
                client=client,
                intent=intent,
                model=args.model,
            )

            print(f"    Reply: {reply!r}")

        except Exception as e:
            print(
                f"    ERROR generating text "
                f"for {intent}: {e}"
            )
            continue

        try:
            print(
                f"    Synthesizing: {out_path}"
            )

            kokoro_to_mp3(
                text=reply,
                out_path=out_path,
                pipeline=pipeline,
                voice=args.voice,
            )

        except Exception as e:
            print(
                f"    ERROR synthesizing "
                f"{intent}: {e}"
            )
            continue

        time.sleep(args.sleep)

    print("Done.")


if __name__ == "__main__":
    main()
