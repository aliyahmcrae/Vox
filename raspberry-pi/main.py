import tomllib
import time
import asyncio
import websockets
import json
import numpy as np
from moonshine_voice import Transcriber, TranscriptEventListener, ModelArch
from typing import Union, Awaitable, Callable, Literal, Any
from collections import deque


class AudioPipeline(TranscriptEventListener):
    text_callback: Callable[[str], Awaitable[None]]
    SAMPLE_RATE: int
    transcriber: Transcriber
    loop: asyncio.AbstractEventLoop

    def __init__(self, audio_config: dict[str, Union[int, float]], text_callback):
        # Initialize queues
        self.SAMPLE_RATE = audio_config["SAMPLE_RATE"]
        self.text_callback = text_callback

        self.transcriber = Transcriber(model_path=audio_config["MODEL_PATH"], model_arch=ModelArch(audio_config["MODEL_ARCH"]))
        self.transcriber.add_listener(self)
        self.transcriber.start()

        self.loop = asyncio.get_event_loop()

    def submit_audio_sample(self, samples):
        self.transcriber.add_audio(samples, self.SAMPLE_RATE)

    def on_line_started(self, event):
        print(f"Line started: {event.line.text}")

    def on_line_text_changed(self, event):
        print(f"Line text changed: {event.line.text}")

    def on_line_completed(self, event):
        print(f"Line completed: {event.line.text}")
        asyncio.run_coroutine_threadsafe(self.text_callback(event.line.text), self.loop)

class QuestionPipeline:
    question_callback: Callable[[str], Awaitable[None]]
    question_queue: asyncio.Queue[str]
    CONTEXT_LENGTH: int

    def __init__(self, config: dict[str, Any], question_callback: Callable[[str], Awaitable[None]]):
        self.question_callback = question_callback
        self.question_queue = asyncio.Queue()

        self.CONTEXT_LENGTH = config["CONTEXT_LENGTH"]

    async def submit_text(self, text: str):
        print(f"[QuestionPipeline] submit_text: queuing text={text!r}")
        await self.question_queue.put(text)

    async def run(self):
        context = deque(maxlen=self.CONTEXT_LENGTH)
        print("[QuestionPipeline] run: started")
        while True:
            text = await self.question_queue.get()
            print(f"[QuestionPipeline] run: dequeued text={text!r}")
            context.append(text)

            # Check for punctuation in the most recent sample. If so, send the last CONTEXT_LENGTH samples onwards
            if any(sep in text for sep in (',', '.', '!', '?')):
                payload = "".join(context)
                print(
                    f"[QuestionPipeline] run: punctuation detected, invoking question_callback with payload={payload!r}")
                await self.question_callback(payload)


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
                print(f"[worker] send_prompt: sending prompt={text!r}")
                await ws.send(json.dumps({
                    "type": "prompt",
                    "data": text
                }))
                print("[worker] send_prompt: send completed")
            except websockets.exceptions.ConnectionClosedOK:
                # remote closed cleanly; ignore this send
                print(
                    "[worker] send_prompt: ConnectionClosedOK while sending; ignoring")
                return
            except Exception as exc:
                # connection closed or other send error; ignore so pipeline can continue/shutdown gracefully
                print(f"[worker] send_prompt: send failed: {exc}")
                return

        question_pipeline = QuestionPipeline(
            conf["question-detection"], send_prompt)
        audio_pipeline = AudioPipeline(
            conf["audio"], question_pipeline.submit_text)

        try:
            print("[worker] sending register_pi")
            await ws.send(json.dumps({
                "type": "register_pi"
            }))
            print("[worker] register_pi sent")
        except websockets.exceptions.ConnectionClosedOK:
            print("Connection closed by server during register; exiting")
            return
        except Exception as e:
            print("Failed to send register message:", e)
            return

        tg.create_task(question_pipeline.run())
        print("[worker] created question_pipeline and audio_pipeline tasks")

        print("Started!")

        async for msg in ws:
            # If the server forwarded raw audio, it will arrive as binary Int16 little-endian
            if isinstance(msg, (bytes, bytearray)):
                samples_i16 = np.frombuffer(msg, dtype=np.int16)
                samples_f32 = samples_i16.astype(np.float32) / 32768.0
                audio_pipeline.submit_audio_sample(samples_f32)
            else:
                try:
                    data = json.loads(msg)
                    print(f"[worker] received text frame: {data}")
                except Exception as e:
                    print(
                        f"[worker] received non-json text frame: {msg!r} error={e}")

if __name__ == "__main__":
    asyncio.run(worker())