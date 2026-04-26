import numpy as np
import sounddevice as sd
from datasets import load_dataset
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import torch

_processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
_model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
_vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

_embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
_speaker_embedding = torch.tensor(_embeddings_dataset[7306]["xvector"]).unsqueeze(0)

SAMPLE_RATE = 16000


def speak(text: str):
    if not text or not text.strip():
        return
    for chunk in _chunk_text(text):
        _play(_synthesize(chunk))


def _synthesize(text: str) -> np.ndarray:
    inputs = _processor(text=text, return_tensors="pt")
    with torch.no_grad():
        speech = _model.generate_speech(inputs["input_ids"], _speaker_embedding, vocoder=_vocoder)
    return speech.numpy()


def _play(audio: np.ndarray):
    sd.play(audio, samplerate=SAMPLE_RATE)
    sd.wait()


def _chunk_text(text: str, max_len: int = 500) -> list[str]:
    if len(text) <= max_len:
        return [text]
    chunks = []
    while len(text) > max_len:
        split_at = text.rfind(". ", 0, max_len)
        if split_at == -1:
            split_at = max_len
        chunks.append(text[:split_at + 1].strip())
        text = text[split_at + 1:].strip()
    if text:
        chunks.append(text)
    return chunks


if __name__ == "__main__":
    speak("Hello! I'm Vox, your voice assistant. How can I help you today?")
