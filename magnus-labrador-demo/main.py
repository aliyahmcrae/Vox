import asyncio
import json
import queue as thread_queue
import random
import signal
import soundfile
import sounddevice as sd
import numpy as np
import noisereduce as nr
import time
import torch
import whisper
from scipy.signal import butter, sosfilt
from threading import Thread, Event
from pathlib import Path

from openai import AsyncOpenAI
from chatterbox.tts import ChatterboxTTS

# Load API key
with open("./labrador/secrets.json") as f:
    secrets = json.load(f)

client = AsyncOpenAI(api_key=secrets["openai"])

# Load Chatterbox TTS model once at startup (runs locally, no API key needed)
_tts_device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading Chatterbox TTS on {_tts_device}...")
tts_model = ChatterboxTTS.from_pretrained(device=_tts_device)
print("Chatterbox TTS ready.")

# Load Whisper STT model — options: tiny / base / small / medium / large
# base: fast, good accuracy; small: slower but noticeably better; medium: best quality
print("Loading Whisper STT model...")
_whisper_model = whisper.load_model("base")
print("Whisper STT ready.")

# Set while TTS/cue audio is playing — mic ignores its own speaker output
_tts_playing = Event()

# Queues
speech_q = asyncio.Queue()
questions_q = asyncio.Queue()
answers_q = asyncio.Queue()
play_q = asyncio.Queue()

CACHE_DIR = Path("./labrador/cues")

SAMPLE_RATE = 16000  # Whisper expects 16 kHz mono
CHUNK_SIZE = 512     # samples per audio callback (~32 ms at 16 kHz)

# VAD thresholds — tune these for your environment
SILENCE_THRESHOLD = 0.01     # mean absolute amplitude below this = silence
SILENCE_DURATION_S = 0.8     # seconds of silence that ends an utterance
MIN_SPEECH_DURATION_S = 0.3  # utterances shorter than this are discarded

ASSISTANT_PROMPT = """You are a voice assistant. Your responses must follow these rules strictly:

* Output plain text only. Do not use markdown, emojis, bullet points, or special formatting.
* Keep responses concise and natural for speech.
* Limit responses to what can be spoken in under 15 seconds (approximately 30–40 words).
* Prioritize clarity and directness over completeness.
* Do not include filler phrases, disclaimers, or unnecessary context.
* Answer the user's question directly. If unsure, say you don't know briefly.
* Avoid lists unless absolutely necessary, and keep them short and spoken naturally.
* Do not repeat the user's question.
* Do not explain your reasoning unless explicitly asked.
* Be warm, friendly, and engaging like a helpful friend.

Speak like a helpful human assistant: brief, clear, and to the point.
"""


def denoise_audio(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    try:
        cleaned = nr.reduce_noise(
            y=audio,
            sr=sr,
            stationary=False,   # handles variable noise (voices, music)
            prop_decrease=0.6,  # less aggressive — over-reduction degrades speech
        )
        return cleaned.astype(np.float32)
    except Exception as e:
        print(f"[noisereduce] Warning: denoising failed, using raw audio. ({e})")
        return audio


def mic_thread_fn(loop, mic_holder):
    SILENCE_CHUNKS = int(SILENCE_DURATION_S * SAMPLE_RATE / CHUNK_SIZE)
    MIN_SPEECH_CHUNKS = int(MIN_SPEECH_DURATION_S * SAMPLE_RATE / CHUNK_SIZE)

    audio_q = thread_queue.Queue()
    active = {"running": True}
    mic_holder["active"] = active

    # VAD state — stored as single-element lists so the nested callback can mutate them
    speech_chunks = []
    silence_count = [0]
    speaking = [False]

    def audio_callback(indata, _frames, _time_info, _status):
        if _tts_playing.is_set():
            speech_chunks.clear()
            silence_count[0] = 0
            speaking[0] = False
            return

        chunk = indata.flatten().astype(np.float32)
        energy = float(np.abs(chunk).mean())

        if energy > SILENCE_THRESHOLD:
            speaking[0] = True
            silence_count[0] = 0
            speech_chunks.append(chunk)
        elif speaking[0]:
            speech_chunks.append(chunk)
            silence_count[0] += 1
            if silence_count[0] >= SILENCE_CHUNKS:
                if len(speech_chunks) >= MIN_SPEECH_CHUNKS:
                    segment = np.concatenate(speech_chunks[:])
                    audio_q.put(segment)
                speech_chunks.clear()
                silence_count[0] = 0
                speaking[0] = False

    def transcribe_worker():
        use_fp16 = torch.cuda.is_available()
        while active["running"]:
            try:
                segment = audio_q.get(timeout=0.5)
            except thread_queue.Empty:
                continue

            cleaned = denoise_audio(segment, SAMPLE_RATE)
            result = _whisper_model.transcribe(
                cleaned,
                language="en",
                fp16=use_fp16,
                initial_prompt="voice assistant command question",
            )
            text = result["text"].strip()
            if text:
                print(f"\n[TRANSCRIPT] {text}")
                asyncio.run_coroutine_threadsafe(speech_q.put(text), loop)

    transcribe_thread = Thread(target=transcribe_worker, daemon=True)
    transcribe_thread.start()

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        blocksize=CHUNK_SIZE,
        dtype="float32",
        callback=audio_callback,
    ):
        while active["running"]:
            time.sleep(0.05)

    transcribe_thread.join(timeout=2.0)


async def question_detector():
    TRIGGERS = {"?", ".", "!"}
    SILENCE_TIMEOUT = 2.0  # seconds of silence after speech to trigger without punctuation
    buffer = ""
    while True:
        try:
            timeout = SILENCE_TIMEOUT if buffer else None
            line = await asyncio.wait_for(speech_q.get(), timeout=timeout)
            print("Detected line:", line)
            buffer = (buffer + " " + line).strip() if buffer else line
            if any(ch in buffer for ch in TRIGGERS):
                await questions_q.put(buffer.strip())
                buffer = ""
        except asyncio.TimeoutError:
            if buffer:
                await questions_q.put(buffer.strip())
                buffer = ""


async def play_random_wav():
    print("Playing cue...")
    if not CACHE_DIR.exists():
        return
    files = [p for p in CACHE_DIR.iterdir() if p.suffix.lower() == ".wav"]
    if not files:
        return
    chosen = random.choice(files)
    print("Chose file!", chosen)

    try:
        # soundfile.read is blocking; run in thread to avoid blocking event loop
        data, sr = await asyncio.to_thread(soundfile.read, str(chosen), dtype="float32")
        _tts_playing.set()
        await asyncio.to_thread(sd.play, data, sr, blocking=True)
    except Exception as e:
        print("Failed to play cue:", e)
    finally:
        _tts_playing.clear()


async def question_handler():
    while True:
        question = await questions_q.get()
        print("Generating response for question:", question)
        # Start the short prompt sound immediately
        play_task = asyncio.create_task(play_random_wav())
        await play_q.put(play_task)

        try:
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": ASSISTANT_PROMPT},
                    {"role": "user", "content": question},
                ],
                max_tokens=100,
            )
            answer = response.choices[0].message.content.strip() or "I don't know."
        except Exception as e:
            print(f"OpenAI error: {e}")
            answer = "I don't know."

        print("Response generated!")
        await answers_q.put(answer)


def _clean_tts_audio(audio: np.ndarray, sr: int) -> np.ndarray:
    # Low-pass filter at 8 kHz — removes high-frequency hiss/static from model artifacts
    sos = butter(6, 8000, btype="low", fs=sr, output="sos")
    audio = sosfilt(sos, audio).astype(np.float32)

    # Normalize to 95% headroom
    max_val = float(np.abs(audio).max())
    if max_val > 0:
        audio = audio / max_val * 0.95

    # Fade in (10 ms) and fade out (40 ms) to eliminate clicks at audio boundaries
    fade_in = int(sr * 0.01)
    fade_out = int(sr * 0.04)
    audio[:fade_in] *= np.linspace(0, 1, fade_in, dtype=np.float32)
    audio[-fade_out:] *= np.linspace(1, 0, fade_out, dtype=np.float32)

    return np.clip(audio, -1.0, 1.0)


async def answer_player():
    while True:
        answer = await answers_q.get()
        try:
            print("Generating audio with Chatterbox...")
            wav = await asyncio.to_thread(tts_model.generate, answer)
            audio = wav.squeeze().numpy()
            audio = _clean_tts_audio(audio, tts_model.sr)
            print("Playing audio...")
            _tts_playing.set()
            try:
                await asyncio.to_thread(sd.play, audio, tts_model.sr, blocking=True)
                await asyncio.sleep(0.3)  # brief cooldown so reverb doesn't get transcribed
            finally:
                _tts_playing.clear()
        except Exception as e:
            print("TTS error:", e)
            _tts_playing.clear()


async def main_async():
    loop = asyncio.get_running_loop()

    mic_holder = {}

    t = Thread(target=mic_thread_fn, args=(loop, mic_holder), daemon=True)
    t.start()

    stop_event = asyncio.Event()

    def _on_stop(*_):
        stop_event.set()

    loop.add_signal_handler(signal.SIGINT, _on_stop)
    loop.add_signal_handler(signal.SIGTERM, _on_stop)

    tasks = [
        asyncio.create_task(question_detector(), name="question_detector"),
        asyncio.create_task(question_handler(), name="question_handler"),
        asyncio.create_task(answer_player(), name="answer_player"),
    ]

    await stop_event.wait()

    active = mic_holder.get("active")
    if active is not None:
        active["running"] = False

    t.join(timeout=2.0)

    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


if __name__ == "__main__":
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        pass
