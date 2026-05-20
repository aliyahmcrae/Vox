from faster_whisper import WhisperModel
import tomllib
import time
import asyncio
import websockets
import json
import numpy as np

from typing import Union, Awaitable, Callable, Literal, Any
from collections import deque


class AudioPipeline:
    SAMPLE_RATE: float
    FRAME_MS: float
    SILENCE_THRESHOLD: float
    BARGE_IN_SPEECH_MS: float
    BARGE_IN_FACTOR: float
    UTTERANCE_DEBOUNCE_MS: float
    SILENCE_MS: float
    SPEECH_MS: float
    CALIBRATE_SECONDS: float
    SPEECH_FACTOR: float

    END_SILENCE_FRAMES: int
    MIN_SPEECH_FRAMES: int
    FRAME_SAMPLES: int

    raw_sample_queue: asyncio.Queue[np.ndarray]
    frame_queue: asyncio.Queue[np.ndarray]
    speech_queue: asyncio.Queue[np.ndarray]

    ambient_energy: list[int] | float
    calibrate_state: Literal["ready", "done"] | float

    model: any
    text_callback: Callable[[str], Awaitable[None]]

    def __init__(self, audio_config: dict[str, Union[int, float]], whisper_model, text_callback):
        self.SAMPLE_RATE = audio_config["SAMPLE_RATE"]
        self.FRAME_MS = audio_config["FRAME_MS"]
        self.SILENCE_THRESHOLD = audio_config["SILENCE_THRESHOLD"]
        self.SILENCE_MS = audio_config["SILENCE_MS"]
        self.SPEECH_MS = audio_config["SPEECH_MS"]
        self.CALIBRATE_SECONDS = audio_config["CALIBRATE_SECONDS"]
        self.SPEECH_FACTOR = audio_config["SPEECH_FACTOR"]

        self.END_SILENCE_FRAMES = int(self.SILENCE_MS / self.FRAME_MS)
        self.MIN_SPEECH_FRAMES = int(self.SPEECH_MS / self.FRAME_MS)
        self.FRAME_SAMPLES = int(self.SAMPLE_RATE * self.FRAME_MS / 1000)

        # Initialize queues
        self.raw_sample_queue = asyncio.Queue()
        self.frame_queue = asyncio.Queue()
        self.speech_queue = asyncio.Queue()

        # calibrate_state == "done" implies ambient_energy is an average of values from time init through CALIBRATE_SECONDS
        self.ambient_energy = []
        self.calibrate_state = "ready"

        self.model = whisper_model
        self.text_callback = text_callback

    async def submit_audio_sample(self, samples: np.ndarray):
        """Accept a 1-D numpy int16 array of samples.

        This uses the samples for ambient noise calibration until calibration completes.
        Once calibrated, pushes the numpy array into raw_sample_queue for framing."""

        if not isinstance(samples, np.ndarray):
            raise TypeError("submit_audio_sample expects a 1-D numpy.ndarray of dtype int16")

        # ensure 1-D
        if samples.ndim != 1:
            samples = samples.flatten()

        # ensure dtype is int16
        if samples.dtype != np.int16:
            samples = samples.astype(np.int16)

        # Have we calibrated yet? If not, set the calibration window
        if self.calibrate_state == "ready":
            self.calibrate_state = time.time() + self.CALIBRATE_SECONDS

        if isinstance(self.calibrate_state, float):
            # still in calibration period: collect absolute magnitudes
            if time.time() < self.calibrate_state or len(self.ambient_energy) == 0:
                if isinstance(self.ambient_energy, list):
                    self.ambient_energy.extend(int(abs(int(x))) for x in samples)
                else:
                    self.ambient_energy = [int(abs(int(x))) for x in samples]
            else:
                # compute ambient energy baseline and finish calibration
                self.ambient_energy = sum(self.ambient_energy) / len(self.ambient_energy) or self.SILENCE_THRESHOLD
                self.calibrate_state = "done"

        # After calibration is done, hand the entire array to the framing queue
        if self.calibrate_state == "done" and samples.size > 0:
            await self.raw_sample_queue.put(samples)

    async def frame_builder(self):
        # maintain a running numpy buffer of int16 samples and emit fixed-size frames
        frame_buffer = np.empty(0, dtype=np.int16)
        while True:
            samples = await self.raw_sample_queue.get()

            # Expect a 1-D numpy int16 array as produced by submit_audio_sample
            if not isinstance(samples, np.ndarray):
                raise TypeError("frame_builder expected numpy.ndarray from raw_sample_queue")
            if samples.ndim != 1:
                samples = samples.flatten()
            if samples.dtype != np.int16:
                samples = samples.astype(np.int16)

            if samples.size == 0:
                continue

            # append incoming samples
            frame_buffer = np.concatenate([frame_buffer, samples])

            # emit as many full frames as possible
            while frame_buffer.size >= self.FRAME_SAMPLES:
                frame = frame_buffer[:self.FRAME_SAMPLES]
                await self.frame_queue.put(frame)
                frame_buffer = frame_buffer[self.FRAME_SAMPLES:]

    async def frame_vad(self):
        vad_buffer = []
        speaking = False
        silence_run = 0
        while True:
            frame = await self.frame_queue.get()
            energy = float(np.abs(frame).mean())
            is_speech = energy > self.ambient_energy * self.SPEECH_FACTOR

            if is_speech:
                vad_buffer.append(frame)
                silence_run = 0

                if len(vad_buffer) >= self.MIN_SPEECH_FRAMES:
                    speaking = True

            elif speaking:  # `and not is_speech`
                vad_buffer.append(frame)
                silence_run += 1

                if silence_run >= self.END_SILENCE_FRAMES:
                    await self.speech_queue.put(np.concatenate(vad_buffer))
                    vad_buffer.clear()
                    silence_run = 0
                    speaking = False

            # Avoids accumulating random noise as "speech"
            elif vad_buffer:
                vad_buffer.clear()

    async def speech_task(self):
        last_reply = ""
        while True:
            speech = await self.speech_queue.get()
            segments, _ = await asyncio.to_thread(
                self.model.transcribe,
                speech,
                language="en",
                initial_prompt=last_reply,
                beam_size=3,
                vad_filter=False
            )
            text = "".join(seg.text for seg in segments).strip()
            if text:
                await self.text_callback(text)
                last_reply = text

    async def run(self):
        async with asyncio.TaskGroup() as tg:
            tg.create_task(self.frame_builder())
            tg.create_task(self.frame_vad())
            tg.create_task(self.speech_task())


class QuestionPipeline:
    question_callback: Callable[[str], Awaitable[None]]
    question_queue: asyncio.Queue[str]
    CONTEXT_LENGTH: int

    def __init__(self, config: dict[str, Any], question_callback: Callable[[str], Awaitable[None]]):
        self.question_callback = question_callback
        self.question_queue = asyncio.Queue()

        self.CONTEXT_LENGTH = config["CONTEXT_LENGTH"]

    async def submit_text(self, text: str):
        await self.question_queue.put(text)

    async def run(self):
        context = deque(maxlen=self.CONTEXT_LENGTH)
        while True:
            text = await self.question_queue.get()
            context.append(text)

            # Check for punctuation in the most recent sample. If so, send the last CONTEXT_LENGTH samples onwards
            if any(sep in text for sep in (',', '.', '!', '?')):
                await self.question_callback("".join(context))


async def worker():
    with open("config.toml") as t:
        conf = tomllib.load(t)

    model = WhisperModel(
        conf["ars"]["model"],
        device="cpu",
        compute_type="int8"
    )

    async with asyncio.TaskGroup() as tg, \
            websockets.connect(
                conf["remote"]["url"],
                max_size=None,
                ping_interval=20) as ws:

        async def send_prompt(text):
            await ws.send(json.dumps({
                "type": "prompt",
                "data": text
            }))

        question_pipeline = QuestionPipeline(
            conf["question-detection"], send_prompt)
        audio_pipeline = AudioPipeline(
            conf["audio"], model, question_pipeline.submit_text)

        await ws.send(json.dumps({
            "type": "register_pi"
        }))

        tg.create_task(question_pipeline.run())
        tg.create_task(audio_pipeline.run())

        async for msg in ws:
            # If the server forwarded raw audio, it will arrive as binary Int16 little-endian
            if isinstance(msg, (bytes, bytearray)):
                samples = np.frombuffer(msg, dtype=np.int16)
                # submit the full array of samples at once (more efficient)
                await audio_pipeline.submit_audio_sample(samples)
                continue

if __name__ == "__main__":
    asyncio.run(worker())
