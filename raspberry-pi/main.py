import tomllib
import time
import asyncio
import json
import numpy as np
import sounddevice as sd
from moonshine_voice import Transcriber, TranscriptEventListener, ModelArch
from typing import Union, Awaitable, Callable


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



async def worker():
    with open("config.toml", "rb") as t:
        conf = tomllib.load(t)

    print("[worker] starting local microphone capture")

    async def print_transcript(text):
        print(f"[worker] transcript: {text!r}")

    # Use a small asyncio.Queue to pass frames out of the real-time audio callback
    # into an asyncio consumer that does any expensive work (resampling / model I/O).
    # This keeps the sounddevice callback fast and reduces input overflow.
    audio_pipeline = AudioPipeline(conf["audio"], print_transcript)

    mic_rate = conf["audio"]["MIC_RATE"]
    sample_rate = conf["audio"]["SAMPLE_RATE"]
    channels = conf["audio"].get("CHANNELS", 1)

    loop = asyncio.get_event_loop()
    audio_queue: asyncio.Queue = asyncio.Queue(maxsize=64)

    # The callback must be as short as possible. We copy the incoming buffer
    # and push it into the asyncio.Queue from the audio thread.
    def audio_callback(indata, frames, time_info, status):
        if status:
            print(f"[audio] status: {status}")
        samples = indata
        # collapse channels in the callback (cheap)
        if samples.ndim > 1:
            samples = samples.mean(axis=1)
        # ensure float32 to avoid dtype conversions later
        if samples.dtype != np.float32:
            samples = samples.astype(np.float32)

        # enqueue a copy; drop frame if the queue is full to avoid blocking
        try:
            loop.call_soon_threadsafe(audio_queue.put_nowait, samples.copy())
        except Exception:
            # QueueFull or other scheduling error; drop frame
            print("[audio] queue full: dropping frame")

    async def audio_consumer():
        # Consumer runs in the asyncio loop and does resampling + submits to model.
        try:
            while True:
                samples = await audio_queue.get()
                try:
                    # simple resampling when mic rate differs from model SAMPLE_RATE
                    if mic_rate != sample_rate:
                        old_len = samples.shape[0]
                        duration = old_len / mic_rate
                        new_len = int(round(duration * sample_rate))
                        if new_len <= 0:
                            continue
                        t_old = np.linspace(0, duration, num=old_len, endpoint=False)
                        t_new = np.linspace(0, duration, num=new_len, endpoint=False)
                        samples = np.interp(t_new, t_old, samples).astype(np.float32)
                    # send to the pipeline (this may do I/O / inference)
                    audio_pipeline.submit_audio_sample(samples)
                finally:
                    audio_queue.task_done()
        except asyncio.CancelledError:
            # gracefully exit on cancellation
            return

    # Choose a small blocksize to keep latency low (e.g. ~10ms)
    blocksize = max(256, int(mic_rate / 100))

    print(f"[worker] opening InputStream mic_rate={mic_rate} sample_rate={sample_rate} blocksize={blocksize}")
    consumer_task = asyncio.create_task(audio_consumer())
    try:
        # request low latency and float32 frames; keep callback minimal
        with sd.InputStream(samplerate=mic_rate, channels=channels, callback=audio_callback, dtype="float32", blocksize=blocksize, latency="low"):
            print("Started! Press Ctrl-C to stop.")
            while True:
                await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("Interrupted, exiting")
    except Exception as e:
        print(f"[worker] audio stream error: {e}")
    finally:
        consumer_task.cancel()
        await asyncio.gather(consumer_task, return_exceptions=True)


if __name__ == "__main__":
    asyncio.run(worker())
