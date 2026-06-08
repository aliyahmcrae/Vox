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

    audio_pipeline = AudioPipeline(conf["audio"], print_transcript)

    mic_rate = conf["audio"]["MIC_RATE"]
    sample_rate = conf["audio"]["SAMPLE_RATE"]
    channels = conf["audio"].get("CHANNELS", 1)

    def audio_callback(indata, frames, time_info, status):
        if status:
            print(f"[audio] status: {status}")
        samples = indata
        if samples.ndim > 1:
            samples = samples.mean(axis=1)

        # simple resampling when mic rate differs from model SAMPLE_RATE
        if mic_rate != sample_rate:
            old_len = samples.shape[0]
            duration = old_len / mic_rate
            new_len = int(round(duration * sample_rate))
            if new_len <= 0:
                return
            t_old = np.linspace(0, duration, num=old_len, endpoint=False)
            t_new = np.linspace(0, duration, num=new_len, endpoint=False)
            samples = np.interp(t_new, t_old, samples).astype(np.float32)
        else:
            samples = samples.astype(np.float32)

        audio_pipeline.submit_audio_sample(samples)

    print(f"[worker] opening InputStream mic_rate={mic_rate} sample_rate={sample_rate}")
    try:
        with sd.InputStream(samplerate=mic_rate, channels=channels, callback=audio_callback):
            print("Started! Press Ctrl-C to stop.")
            while True:
                await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("Interrupted, exiting")
    except Exception as e:
        print(f"[worker] audio stream error: {e}")


if __name__ == "__main__":
    asyncio.run(worker())
