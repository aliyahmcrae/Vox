import tomllib
import time
import asyncio
import websockets
import json
import numpy as np
from moonshine_voice import Transcriber, TranscriptEventListener, ModelArch
from typing import Union, Awaitable, Callable, Any


class AudioPipeline(TranscriptEventListener):
    text_callback: Callable[[str], Awaitable[None]]
    SAMPLE_RATE: int
    transcriber: Transcriber
    loop: asyncio.AbstractEventLoop

    def __init__(self, audio_config: dict[str, Union[int, float]], text_callback):
        self.SAMPLE_RATE = audio_config["SAMPLE_RATE"]
        self.text_callback = text_callback

        self.transcriber = Transcriber(
            model_path=audio_config["MODEL_PATH"],
            model_arch=ModelArch(audio_config["MODEL_ARCH"])
        )
        self.transcriber.add_listener(self)
        self.transcriber.start()

        self.loop = asyncio.get_event_loop()

    def submit_audio_sample(self, samples):
        self.transcriber.add_audio(samples, self.SAMPLE_RATE)

    def on_line_started(self, event):
        print(f"[ASR] line_started: {event.line.text!r}")

    def on_line_text_changed(self, event):
        print(f"[ASR] line_changed: {event.line.text!r}")

    def on_line_completed(self, event):
        # CHANGE 1: on_line_completed is the primary turn-end signal.
        # We hand it to the pipeline, which will debounce before sending.
        print(f"[ASR] line_completed: {event.line.text!r}")
        asyncio.run_coroutine_threadsafe(
            self.text_callback(event.line.text), self.loop
        )


class QuestionPipeline:
    question_callback: Callable[[str], Awaitable[None]]
    question_queue: asyncio.Queue[str]
    DEBOUNCE_SECONDS: float

    # CHANGE 2: Removed deque / CONTEXT_LENGTH entirely.
    # CHANGE 3: Added pending_text + last_text_time for silence-based endpointing.
    def __init__(self, config: dict[str, Any], question_callback: Callable[[str], Awaitable[None]]):
        self.question_callback = question_callback
        self.question_queue = asyncio.Queue()

        # Tune this in config.toml — 0.5 is snappy, 0.8 is safer
        self.DEBOUNCE_SECONDS = config.get("DEBOUNCE_SECONDS", 0.6)

        self.pending_text = ""
        self.last_text_time = 0.0
        self.is_processing = False  # add this

    async def submit_text(self, text: str):
        print(f"[QuestionPipeline] queuing: {text!r}")
        await self.question_queue.put(text)

    async def run(self):
        print("[QuestionPipeline] run: started")
        while True:
            text = await self.question_queue.get()
            print(f"[QuestionPipeline] received: {text!r}")

            if not text.strip():
                continue

            if self.pending_text:
                self.pending_text = self.pending_text + " " + text
            else:
                self.pending_text = text
            
            self.last_text_time = time.time()

    
    async def endpoint_detector(self):
        print("[QuestionPipeline] endpoint_detector: started")
        while True:
            await asyncio.sleep(0.1)

            if not self.pending_text or self.is_processing:
                continue

            silence = time.time() - self.last_text_time

            if silence > self.DEBOUNCE_SECONDS:
                payload = self.pending_text.strip()
    
                if len(payload.split()) < 2:  # ignore anything under 2 words
                    self.pending_text = ""
                    continue

                self.is_processing = True
                print(f"[QuestionPipeline] endpoint: silence={silence:.2f}s — sending: {payload!r}")
                print(f"[TIMING] speech_end → request_sent: {time.time():.3f}")
                await self.question_callback(payload)
                await asyncio.sleep(5)  # block new prompts for 5s while assistant responds
                self.is_processing = False
                self.pending_text = ""  # discard anything that accumulated during response


async def worker():
    with open("config.toml", "rb") as t:
        conf = tomllib.load(t)

    print(f"[worker] connecting to {conf['remote']['url']}")
    async with asyncio.TaskGroup() as tg, \
            websockets.connect(
                conf["remote"]["url"],
                max_size=None,
                ping_interval=20) as ws:

        print("[worker] websocket connected")

        async def send_prompt(text):
            try:
                print(f"[worker] send_prompt: {text!r}")
                # TIMING LOG 2: request_sent timestamp
                print(f"[TIMING] request_sent: {time.time():.3f}")
                await ws.send(json.dumps({
                    "type": "prompt",
                    "data": text
                }))
                print("[worker] send_prompt: done")
            except websockets.exceptions.ConnectionClosedOK:
                print("[worker] send_prompt: connection closed cleanly, ignoring")
            except Exception as exc:
                print(f"[worker] send_prompt: failed: {exc}")

        question_pipeline = QuestionPipeline(conf["question-detection"], send_prompt)
        audio_pipeline = AudioPipeline(conf["audio"], question_pipeline.submit_text)

        try:
            print("[worker] sending register_pi")
            await ws.send(json.dumps({"type": "register_pi"}))
            print("[worker] register_pi sent")
        except websockets.exceptions.ConnectionClosedOK:
            print("[worker] connection closed during register; exiting")
            return
        except Exception as e:
            print("[worker] failed to send register:", e)
            return

        # CHANGE 6: Create both tasks — run() and endpoint_detector() run concurrently
        tg.create_task(question_pipeline.run())
        tg.create_task(question_pipeline.endpoint_detector())
        print("[worker] tasks created")
        print("Started!")

        async for msg in ws:
            if isinstance(msg, (bytes, bytearray)):
                samples_i16 = np.frombuffer(msg, dtype=np.int16)
                samples_f32 = samples_i16.astype(np.float32) / 32768.0
                audio_pipeline.submit_audio_sample(samples_f32)
            else:
                try:
                    data = json.loads(msg)
                    print(f"[worker] text frame: {data}")
                except Exception as e:
                    print(f"[worker] non-json frame: {msg!r} error={e}")


if __name__ == "__main__":
    asyncio.run(worker())