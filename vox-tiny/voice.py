#!/usr/bin/env python3
"""
Generate short spoken MP3 responses for each intent.

Usage:
  python3 voice.py            # uses vox-tiny/intent_names.txt, writes to ./intents/
  python3 voice.py --force    # regenerate all mp3s even if they exist

Requires:
  - OPENAI_API_KEY environment variable
  - pip install openai gtts

The script:
- loads unique intent names from vox-tiny/intent_names.txt
- for each intent asks the GPT model for a short response (using the
  model gpt-realtime-mini-2025-12-15 by default)
- synthesizes the returned text to an mp3 using gTTS and saves intents/{intent}.mp3
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

try:
    import openai
except Exception as e:
    print("Missing dependency: openai. Install with: pip install openai")
    raise

try:
    from gtts import gTTS
except Exception as e:
    print("Missing dependency: gtts. Install with: pip install gtts")
    raise


DEFAULT_INTENT_FILE = Path("intent_names.txt")
DEFAULT_OUT_DIR = Path("intents")
DEFAULT_MODEL = "gpt-5.4-mini"


EXAMPLES_TEXT = """Examples:
Listening...
'help me buy a train ticket to boston'
→ transport_ticket (0.755)
Would've played: "Ok, I'll check prices for you!"

'help me create a new list'
→ lists_createoradd (0.825)
Would've played: intents/lists_createoradd.mp3 "Got it! I'll write that down."

'turn off the lights please'
→ iot_hue_lightoff (0.859)
Would've played: intents/iot_hue_lightoff.mp3 "On it! Dimming the lights..."
"""

SYSTEM_INSTRUCTION = (
    "You are a concise assistant that writes short, polite, natural spoken replies "
    "for a voice assistant. For a given intent name produce a short spoken response "
    "that a generic assistant could say out loud. Keep replies short (about 2-10 words), "
    "natural, and unambiguous. Output only the reply text, without surrounding quotes, "
    "metadata, or explanations."
)

def create_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY not set in environment"
        )

    return openai.OpenAI(api_key=api_key)

def load_intents(intent_file: Path) -> list[str]:
    if not intent_file.exists():
        raise FileNotFoundError(f"Intent file not found: {intent_file}")
    with intent_file.open("r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    # preserve order and deduplicate (first occurrence wins)
    seen = set()
    unique = []
    for l in lines:
        if l not in seen:
            seen.add(l)
            unique.append(l)
    return unique

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

def synthesize_to_mp3(text: str, out_path: Path, lang: str = "en"):
    """
    Uses gTTS to synthesize text to an mp3 file at out_path.
    """
    if not text:
        raise ValueError("Empty text provided for synthesis")
    tts = gTTS(text=text, lang=lang)
    tmp = out_path.with_suffix(".tmp.mp3")
    tts.save(str(tmp))
    tmp.rename(out_path)


def main():
    p = argparse.ArgumentParser(description="Generate mp3 responses for intents.")
    p.add_argument("--intents-file", default=str(DEFAULT_INTENT_FILE))
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--force", action="store_true", help="Regenerate MP3 files even if they exist")
    p.add_argument("--sleep", type=float, default=1.0, help="Seconds to sleep between OpenAI requests")
    args = p.parse_args()

    intents_file = Path(args.intents_file)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        intents = load_intents(intents_file)
    except Exception as e:
        print(f"Error loading intents: {e}")
        sys.exit(1)

    print(f"Loaded {len(intents)} unique intents (will write to {out_dir}).")

    client = create_client()

    for i, intent in enumerate(intents, start=1):
        safe_name = intent.replace("/", "_").replace(" ", "_")
        out_path = out_dir / f"{safe_name}.mp3"
        if out_path.exists() and not args.force:
            print(f"[{i}/{len(intents)}] Skipping existing: {out_path}")
            continue

        try:
            print(f"[{i}/{len(intents)}] Asking model for intent: {intent}")
            reply = ask_gpt_for_reply(client, intent, model=args.model)
            print(f"  -> Reply: {reply!r}")
        except Exception as e:
            print(f"  ERROR asking model for {intent}: {e}")
            continue

        try:
            print(f"  Synthesizing to {out_path} ...")
            synthesize_to_mp3(reply, out_path)
        except Exception as e:
            print(f"  ERROR synthesizing {intent}: {e}")
            continue

        # polite pause to avoid rate-limits
        time.sleep(args.sleep)

    print("Done.")


if __name__ == "__main__":
    main()
