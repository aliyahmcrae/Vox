import sounddevice as sd
import numpy as np
from transformers import pipeline

SAMPLE_RATE = 16000
RECORD_SECONDS = 5

_stt = pipeline("automatic-speech-recognition", model="openai/whisper-base")


def record_audio(duration: int = RECORD_SECONDS) -> np.ndarray:
    audio = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    return audio.flatten()


def transcribe_array(audio: np.ndarray) -> str:
    result = _stt({"sampling_rate": SAMPLE_RATE, "raw": audio})
    return result["text"].strip()


def transcribe_file(path: str) -> str:
    result = _stt(path)
    return result["text"].strip()


def listen_and_transcribe(duration: int = RECORD_SECONDS) -> str:
    audio = record_audio(duration)
    return transcribe_array(audio)


if __name__ == "__main__":
    text = listen_and_transcribe()
    print(f"Transcribed: {text}")
