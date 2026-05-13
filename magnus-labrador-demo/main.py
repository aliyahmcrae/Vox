import asyncio
import json
import os
import queue as thread_queue
import random
import signal
import soundfile
import sounddevice as sd
import numpy as np
import noisereduce as nr
import time
from threading import Thread, Event
from pathlib import Path

from openai import AsyncOpenAI
from faster_whisper import WhisperModel
from piper.voice import PiperVoice

PROJECT_DIR = Path(__file__).parent

with open(PROJECT_DIR / "labrador" / "secrets.json") as f:
    secrets = json.load(f)

client = AsyncOpenAI(api_key=secrets["openai"])

# TTS: Piper, CPU-only. Swap voice via PIPER_VOICE env var.
PIPER_VOICE_PATH = PROJECT_DIR / "labrador" / os.environ.get(
    "PIPER_VOICE", "en_US-lessac-medium.onnx"
)
print(f"Loading Piper TTS voice from {PIPER_VOICE_PATH.name}...")
tts_voice = PiperVoice.load(str(PIPER_VOICE_PATH))
TTS_SAMPLE_RATE = tts_voice.config.sample_rate
print(f"Piper TTS ready ({TTS_SAMPLE_RATE} Hz).")

# STT: faster-whisper, int8 on CPU. tiny.en for Pi, base.en if you have headroom.
WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL", "tiny.en")
print(f"Loading faster-whisper STT model ({WHISPER_MODEL_SIZE})...")
_whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
print("faster-whisper STT ready.")

# Set while we're playing audio — mic ignores its own output.
# Refcounted because cue + answer playback can overlap.
_tts_playing = Event()
_playing_refs = 0


def _start_playing():
    global _playing_refs
    _playing_refs += 1
    _tts_playing.set()


def _stop_playing():
    global _playing_refs
    _playing_refs = max(0, _playing_refs - 1)
    if _playing_refs == 0:
        _tts_playing.clear()


speech_q = asyncio.Queue()
questions_q = asyncio.Queue()
answers_q = asyncio.Queue()

# Hold task refs so asyncio doesn't GC them mid-flight.
_bg_tasks: set[asyncio.Task] = set()

CACHE_DIR = Path("./labrador/cues")

SAMPLE_RATE = 16000  # whisper wants 16k mono
CHUNK_SIZE = 512     # ~32 ms per callback

# VAD — tune for your room.
SILENCE_THRESHOLD = 0.01
SILENCE_DURATION_S = 0.8
MIN_SPEECH_DURATION_S = 0.3

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
            stationary=False,
            prop_decrease=0.6,  # cranking this higher destroys speech
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

    # lists so the audio_callback closure can mutate them
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
        while active["running"]:
            try:
                segment = audio_q.get(timeout=0.5)
            except thread_queue.Empty:
                continue

            cleaned = denoise_audio(segment, SAMPLE_RATE)
            segments, _info = _whisper_model.transcribe(
                cleaned,
                language="en",
                initial_prompt="voice assistant command question",
                beam_size=1,  # greedy, faster on CPU
                vad_filter=False,  # already VAD'd upstream
            )
            text = "".join(seg.text for seg in segments).strip()
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
    SILENCE_TIMEOUT = 2.0  # fallback flush if whisper drops punctuation
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

    _start_playing()
    try:
        # soundfile.read blocks — keep it off the event loop
        data, sr = await asyncio.to_thread(soundfile.read, str(chosen), dtype="float32")
        await asyncio.to_thread(sd.play, data, sr, blocking=True)
    except Exception as e:
        print("Failed to play cue:", e)
    finally:
        _stop_playing()


async def question_handler():
    while True:
        question = await questions_q.get()
        print("Generating response for question:", question)
        # filler cue while the LLM cooks
        play_task = asyncio.create_task(play_random_wav())
        _bg_tasks.add(play_task)
        play_task.add_done_callback(_bg_tasks.discard)

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
    # fades kill the click at start/end
    fade_in = int(sr * 0.01)
    fade_out = int(sr * 0.04)
    if fade_in > 0:
        audio[:fade_in] *= np.linspace(0, 1, fade_in, dtype=np.float32)
    if fade_out > 0:
        audio[-fade_out:] *= np.linspace(1, 0, fade_out, dtype=np.float32)
    return np.clip(audio, -1.0, 1.0)


def _piper_synthesize(text: str) -> np.ndarray:
    # piper 1.x: synthesize() yields AudioChunk objects with float arrays per sentence
    chunks = [chunk.audio_float_array for chunk in tts_voice.synthesize(text)]
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    pcm = np.concatenate(chunks).astype(np.float32)
    return _clean_tts_audio(pcm, TTS_SAMPLE_RATE)


async def answer_player():
    while True:
        answer = await answers_q.get()
        try:
            print("Generating audio with Piper...")
            audio = await asyncio.to_thread(_piper_synthesize, answer)
            print("Playing audio...")
            _start_playing()
            try:
                await asyncio.to_thread(sd.play, audio, TTS_SAMPLE_RATE, blocking=True)
                await asyncio.sleep(0.3)  # let room reverb die before reopening mic
            finally:
                _stop_playing()
        except Exception as e:
            print("TTS error:", e)


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
