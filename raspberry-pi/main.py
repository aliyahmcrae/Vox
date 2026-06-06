import tomllib
import time
import asyncio
import websockets
import json
import numpy as np
from moonshine_voice import Transcriber, TranscriptEventListener, ModelArch
from typing import Union, Awaitable, Callable


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

        async def send_transcript(text):
            try:
                print(f"[worker] send_transcript: sending transcript={text!r}")
                await ws.send(json.dumps({
                    "type": "transcript",
                    "data": text
                }))
                print("[worker] send_transcript: send completed")
            except websockets.exceptions.ConnectionClosedOK:
                # remote closed cleanly; ignore this send
                print(
                    "[worker] send_transcript: ConnectionClosedOK while sending; ignoring")
                return
            except Exception as exc:
                # connection closed or other send error; ignore so pipeline can continue/shutdown gracefully
                print(f"[worker] send_transcript: send failed: {exc}")
                return

        # The cloud now handles intent detection, so the Pi just streams each
        # completed transcript line as the model produces it.
        audio_pipeline = AudioPipeline(conf["audio"], send_transcript)

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

        print("[worker] created audio_pipeline")

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