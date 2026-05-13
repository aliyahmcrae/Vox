import asyncio
import json
import os
import time
import queue as thread_queue
from threading import Thread

import numpy as np
import noisereduce as nr
import websockets
from faster_whisper import WhisperModel


# -------------------------
# Config
# -------------------------

ORACLE_WS = os.environ.get(
    "ORACLE_WS",
    "wss://api.magnusfulton.com/cse481/ws/pi"
)

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 2048

SILENCE_THRESHOLD = 0.01
SILENCE_DURATION_S = 0.8
MIN_SPEECH_DURATION_S = 0.3

WHISPER_MODEL = os.environ.get(
    "WHISPER_MODEL",
    "tiny.en"
)

SILENCE_CHUNKS = int(
    SILENCE_DURATION_S * SAMPLE_RATE / CHUNK_SAMPLES
)

MIN_SPEECH_CHUNKS = int(
    MIN_SPEECH_DURATION_S * SAMPLE_RATE / CHUNK_SAMPLES
)


# -------------------------
# Whisper
# -------------------------

print(f"loading whisper: {WHISPER_MODEL}")
model = WhisperModel(
    WHISPER_MODEL,
    device="cpu",
    compute_type="int8"
)
print("whisper ready")


# -------------------------
# Denoise
# -------------------------

def denoise(audio):
    try:
        return nr.reduce_noise(
            y=audio,
            sr=SAMPLE_RATE,
            stationary=False,
            prop_decrease=0.6
        ).astype(np.float32)
    except Exception:
        return audio


# -------------------------
# Question detector
# -------------------------

class QuestionDetector:
    def __init__(self):
        self.buffer = ""

    def feed(self, text):
        self.buffer = (
            f"{self.buffer} {text}".strip()
            if self.buffer else text
        )

        if any(
            x in self.buffer
            for x in ["?", ".", "!"]
        ):
            out = self.buffer
            self.buffer = ""
            return out

        return None


detector = QuestionDetector()


# -------------------------
# Audio segmenter
# -------------------------

class AudioPipeline:
    def __init__(self, loop, ws):
        self.loop = loop
        self.ws = ws

        self.audio_q = thread_queue.Queue()

        self.speaking = False
        self.silence_count = 0
        self.chunks = []

        self.worker = Thread(
            target=self.transcribe_worker,
            daemon=True
        )
        self.worker.start()

    def push_pcm(self, pcm_bytes):
        pcm = np.frombuffer(
            pcm_bytes,
            dtype=np.int16
        ).astype(np.float32) / 32768.0

        energy = float(np.abs(pcm).mean())

        if energy > SILENCE_THRESHOLD:
            self.speaking = True
            self.silence_count = 0
            self.chunks.append(pcm)

        elif self.speaking:
            self.chunks.append(pcm)
            self.silence_count += 1

            if self.silence_count >= SILENCE_CHUNKS:
                if len(self.chunks) >= MIN_SPEECH_CHUNKS:
                    segment = np.concatenate(
                        self.chunks
                    )
                    self.audio_q.put(segment)

                self.chunks.clear()
                self.speaking = False
                self.silence_count = 0

    def transcribe_worker(self):
        while True:
            segment = self.audio_q.get()

            cleaned = denoise(segment)

            segments, _ = model.transcribe(
                cleaned,
                language="en",
                beam_size=1,
                vad_filter=False,
                initial_prompt=(
                    "voice assistant question"
                )
            )

            text = "".join(
                s.text for s in segments
            ).strip()

            if not text:
                continue

            print("transcript:", text)

            q = detector.feed(text)

            if q:
                print("question:", q)

                asyncio.run_coroutine_threadsafe(
                    self.ws.send(
                        json.dumps({
                            "type": "question",
                            "text": q
                        })
                    ),
                    self.loop
                )


# -------------------------
# Main websocket loop
# -------------------------

async def worker():
    while True:
        try:
            async with websockets.connect(
                ORACLE_WS,
                max_size=None,
                ping_interval=20
            ) as ws:

                print("connected")

                loop = asyncio.get_running_loop()
                pipe = AudioPipeline(
                    loop,
                    ws
                )

                await ws.send(json.dumps({
                    "type": "register_pi"
                }))

                async for msg in ws:
                    if isinstance(msg, bytes):
                        pipe.push_pcm(msg)

        except Exception as e:
            print("disconnected:", e)

        await asyncio.sleep(2)


if __name__ == "__main__":
    asyncio.run(worker())