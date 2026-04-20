import asyncio, json
import hashlib
from pathlib import Path

from openai import AsyncOpenAI
from openai.helpers import LocalAudioPlayer

with open("secrets.json") as f:
    secrets = json.load(f)

client = AsyncOpenAI(api_key=secrets["openai"])

CACHE_DIR = Path("./tts_cache")
CACHE_DIR.mkdir(exist_ok=True)

def cache_key(text: str) -> Path:
    h = hashlib.md5(text.encode()).hexdigest()
    return CACHE_DIR / f"{h}.wav"

async def speak(text: str):
    path = cache_key(text)

    # If cached, just play it
    if path.exists():
        print("[cache hit]")
        #await LocalAudioPlayer().play_file(path)
        return

    print("[generating]")

    async with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="alloy",
        input=text,
        response_format="wav",  # low latency playback
    ) as response:

        # Save while streaming
        with open(path, "wb") as f:
            async for chunk in response.iter_bytes():
                f.write(chunk)

        # Play after save (simple version)
        #await LocalAudioPlayer().play_file(path)


async def main():
    await speak("Hmm, let me think about that!")
    await speak("Let me look something up really quick...")
    await speak("Ooo, great question! One second...")


if __name__ == "__main__":
    asyncio.run(main())
