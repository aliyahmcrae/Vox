import asyncio
import json
import os
from pathlib import Path

import websockets
from openai import AsyncOpenAI
from piper.voice import PiperVoice


OPENAI_KEY = os.environ["OPENAI_API_KEY"]

client = AsyncOpenAI(api_key=OPENAI_KEY)

PIPER_PATH = Path("./en_US-lessac-medium.onnx")
tts = PiperVoice.load(str(PIPER_PATH))
TTS_SR = tts.config.sample_rate


ASSISTANT_PROMPT = """
You are a voice assistant.

Plain text only.
Brief.
Natural speech.
Under 40 words.
Answer directly.
"""


pi_socket = None
client_socket = None


def synthesize(text: str):
    chunks = [c.audio_float_array for c in tts.synthesize(text)]
    if not chunks:
        return b""

    import numpy as np

    pcm = np.concatenate(chunks)
    pcm16 = (pcm * 32767).astype(np.int16)

    return pcm16.tobytes()


async def generate_answer(question: str):
    try:
        r = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": ASSISTANT_PROMPT},
                {"role": "user", "content": question},
            ],
            max_tokens=100,
        )
        return r.choices[0].message.content.strip()
    except Exception:
        return "I don't know."


async def handle_question(question):
    global client_socket

    if not client_socket:
        return

    await client_socket.send(
        json.dumps({"type": "answering"})
    )

    answer = await generate_answer(question)

    audio = await asyncio.to_thread(
        synthesize,
        answer
    )

    CHUNK = 4096

    for i in range(0, len(audio), CHUNK):
        await client_socket.send(audio[i:i+CHUNK])
        await asyncio.sleep(0)

    await client_socket.send(
        json.dumps({"type": "done"})
    )


async def handle_pi(ws):
    global pi_socket
    pi_socket = ws

    print("pi connected")

    try:
        async for msg in ws:
            if isinstance(msg, bytes):
                continue

            data = json.loads(msg)

            if data["type"] == "question":
                asyncio.create_task(
                    handle_question(data["text"])
                )

    finally:
        pi_socket = None
        print("pi disconnected")


async def handle_client(ws):
    global client_socket
    client_socket = ws

    print("client connected")

    try:
        async for msg in ws:
            if pi_socket:
                await pi_socket.send(msg)

    finally:
        client_socket = None
        print("client disconnected")


async def router(ws):
    path = ws.request.path

    if path == "/ws/pi":
        await handle_pi(ws)

    elif path == "/ws/client":
        await handle_client(ws)

    else:
        await ws.close()


async def main():
    async with websockets.serve(
        router,
        "0.0.0.0",
        8765,
        max_size=None,
        ping_interval=20
    ):
        print("oracle relay up")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())