import asyncio
import json
import os
import queue as thread_queue
import re
import signal
import time
from pathlib import Path
from threading import Event, Thread

import numpy as np
import sounddevice as sd

from openai import AsyncOpenAI
from faster_whisper import WhisperModel
from piper.voice import PiperVoice

PROJECT_DIR = Path(__file__).parent
LABRADOR_DIR = PROJECT_DIR / "labrador"
PROFILE_PATH = LABRADOR_DIR / "profile.json"
HISTORY_PATH = LABRADOR_DIR / "history.jsonl"

# Set BARGE_IN=0 to disable mid-speech interruption if your speaker/mic setup
# has feedback or the open-air echo path keeps triggering false barge-ins.
BARGE_IN = os.environ.get("BARGE_IN", "1") == "1"

with open(LABRADOR_DIR / "secrets.json") as f:
    secrets = json.load(f)

client = AsyncOpenAI(api_key=secrets["openai"])


# ----- Persistent profile + cross-session history -----

def load_profile() -> dict:
    if PROFILE_PATH.exists():
        try:
            return json.loads(PROFILE_PATH.read_text())
        except Exception:
            pass
    return {"name": None, "preferences": []}


def load_recent_history(n: int) -> list[dict]:
    if not HISTORY_PATH.exists():
        return []
    try:
        lines = HISTORY_PATH.read_text().splitlines()[-n:]
        return [json.loads(l) for l in lines if l.strip()]
    except Exception:
        return []


def append_history(message: dict) -> None:
    try:
        with open(HISTORY_PATH, "a") as f:
            f.write(json.dumps(message) + "\n")
    except Exception as e:
        print(f"[history] write failed: {e}")


profile = load_profile()


# ----- Models -----

PIPER_VOICE_PATH = LABRADOR_DIR / os.environ.get(
    "PIPER_VOICE", "en_US-amy-medium.onnx"
)
print(f"Loading Piper TTS voice from {PIPER_VOICE_PATH.name}...")
tts_voice = PiperVoice.load(str(PIPER_VOICE_PATH))
TTS_SAMPLE_RATE = tts_voice.config.sample_rate
print(f"Piper TTS ready ({TTS_SAMPLE_RATE} Hz).")

WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL", "base.en")
print(f"Loading faster-whisper STT model ({WHISPER_MODEL_SIZE})...")
_whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
print("faster-whisper STT ready.")


# ----- Audio / VAD config -----

SAMPLE_RATE = 16000
FRAME_MS = 20
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000  # 320 samples per frame

# Simple energy threshold for speech detection (can be overridden by env)
SILENCE_THRESHOLD = float(os.environ.get("SILENCE_THRESHOLD", "0.02"))
# Calibrate barge-in sensitivity (ms of continuous speech required to interrupt TTS)
BARGE_IN_SPEECH_MS = int(os.environ.get("BARGE_IN_SPEECH_MS", "350"))
BARGE_IN_SPEECH_FRAMES = int(BARGE_IN_SPEECH_MS / FRAME_MS)
# When calibrating against ambient noise, require speech energy to exceed ambient * this factor
BARGE_IN_FACTOR = float(os.environ.get("BARGE_IN_FACTOR", "3.0"))

END_SILENCE_FRAMES = int(700 / FRAME_MS)    # 700 ms of silence ends utterance
MIN_SPEECH_FRAMES = int(300 / FRAME_MS)     # need 300 ms of speech to count

UTTERANCE_DEBOUNCE_S = 0.5
MAX_HISTORY_MESSAGES = 10

# Load recent turns so context survives restarts.
conversation_history: list[dict] = load_recent_history(MAX_HISTORY_MESSAGES)
_last_assistant_reply: str = next(
    (m["content"] for m in reversed(conversation_history) if m.get("role") == "assistant"),
    "",
)


# ----- TTS playback state (refcounted) -----

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


# ----- Queues + barge-in coordination -----

speech_q: asyncio.Queue = asyncio.Queue()      # raw STT fragments
questions_q: asyncio.Queue = asyncio.Queue()   # debounced complete utterances


class ResponseSession:
    """One LLM-stream-to-speech response. Barge-in sets cancel_event."""

    def __init__(self):
        self.cancel_event = asyncio.Event()
        self.sentence_q: asyncio.Queue = asyncio.Queue()  # str | None
        self.audio_q: asyncio.Queue = asyncio.Queue()      # np.ndarray | None
        self.full_answer = ""


active_response: ResponseSession | None = None


def _request_barge_in(loop: asyncio.AbstractEventLoop) -> None:
    """Called from the mic thread when the user speaks during TTS."""
    resp = active_response
    if resp is None or resp.cancel_event.is_set():
        return
    loop.call_soon_threadsafe(resp.cancel_event.set)
    try:
        sd.stop()
    except Exception:
        pass
    print("[BARGE-IN] User interrupted.")


# ----- System prompt with profile -----

BASE_PROMPT = """You are Vox, a voice assistant in a real-time spoken conversation. You sound like a person, not a chatbot.

Style:
* Plain spoken text only. No markdown, bullets, emojis, or lists.
* Keep replies to one short sentence (under 20 words) whenever possible. Two only if truly needed.
* Use contractions and natural phrasing. Speak the way a person would, not a written paragraph.
* Never preface with "Sure", "Okay", "Got it", "Hmm", "Let me", or any acknowledgment filler. Start with the answer.
* Do not repeat or paraphrase the question back.

Content:
* Answer the actual thing asked. Don't dump background, caveats, or extra detail unless asked.
* If the user asks several things at once, answer them all in one tight reply.
* Use prior turns for context — resolve pronouns and follow-ups naturally without asking.
* Only ask a clarifying question if the request is genuinely ambiguous and you can't reasonably guess. Otherwise make an assumption and answer.
* If you don't know, say so briefly in one sentence.
"""


def build_system_prompt() -> str:
    out = BASE_PROMPT
    if profile.get("name"):
        out += f"\nThe user's name is {profile['name']}."
    prefs = profile.get("preferences") or []
    if prefs:
        bullets = "\n".join(f"- {p}" for p in prefs)
        out += f"\nKnown about the user:\n{bullets}"
    return out


# ----- Mic thread: capture → energy-based VAD → Whisper -----

def mic_thread_fn(loop: asyncio.AbstractEventLoop, mic_holder: dict) -> None:
    active = {"running": True}
    mic_holder["active"] = active

    frame_q: thread_queue.Queue = thread_queue.Queue()
    transcribe_q: thread_queue.Queue = thread_queue.Queue()

    # Calibrate ambient noise to reduce false barge-ins. We sample a short buffer
    # and set a dynamic threshold = max(SILENCE_THRESHOLD, ambient * BARGE_IN_FACTOR).
    CALIBRATE_SECONDS = float(os.environ.get("CALIBRATE_SECONDS", "0.5"))
    try:
        rec = sd.rec(int(CALIBRATE_SECONDS * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32")
        sd.wait(timeout=CALIBRATE_SECONDS + 1)
        ambient_energy = float(np.abs(rec).mean())
    except Exception:
        ambient_energy = SILENCE_THRESHOLD
    dynamic_threshold = max(SILENCE_THRESHOLD, ambient_energy * BARGE_IN_FACTOR)
    print(f"[VAD] ambient_energy={ambient_energy:.6f}, threshold={dynamic_threshold:.6f}")

    speech_frames: list[np.ndarray] = []
    silence_run = [0]
    speaking = [False]
    bargein_run = [0]

    def audio_callback(indata, _frames, _time_info, _status):
        frame_q.put(indata.flatten().astype(np.float32, copy=True))

    def vad_worker():
        while active["running"]:
            try:
                frame = frame_q.get(timeout=0.5)
            except thread_queue.Empty:
                continue

            # Simple energy-based speech detection
            energy = float(np.abs(frame).mean())
            is_speech = energy > dynamic_threshold

            tts_active = _tts_playing.is_set()

            if tts_active and not speaking[0]:
                # During TTS playback, require sustained speech to interrupt
                if BARGE_IN and is_speech:
                    bargein_run[0] += 1
                    if bargein_run[0] >= BARGE_IN_SPEECH_FRAMES:
                        _request_barge_in(loop)
                        bargein_run[0] = 0
                        speech_frames.clear()
                        silence_run[0] = 0
                        speaking[0] = True
                        speech_frames.append(frame)
                else:
                    bargein_run[0] = 0
                continue

            bargein_run[0] = 0

            if is_speech:
                speaking[0] = True
                silence_run[0] = 0
                speech_frames.append(frame)
            elif speaking[0]:
                speech_frames.append(frame)
                silence_run[0] += 1
                if silence_run[0] >= END_SILENCE_FRAMES:
                    if len(speech_frames) >= MIN_SPEECH_FRAMES:
                        transcribe_q.put(np.concatenate(speech_frames))
                    speech_frames.clear()
                    silence_run[0] = 0
                    speaking[0] = False

    def transcribe_worker():
        while active["running"]:
            try:
                segment = transcribe_q.get(timeout=0.5)
            except thread_queue.Empty:
                continue

            prompt = _last_assistant_reply or "voice assistant command question"
            segments, _info = _whisper_model.transcribe(
                segment,
                language="en",
                initial_prompt=prompt,
                beam_size=3,
                vad_filter=False,
            )
            text = "".join(seg.text for seg in segments).strip()
            if text:
                print(f"\n[TRANSCRIPT] {text}")
                asyncio.run_coroutine_threadsafe(speech_q.put(text), loop)

    vad_thread = Thread(target=vad_worker, daemon=True)
    vad_thread.start()
    tr_thread = Thread(target=transcribe_worker, daemon=True)
    tr_thread.start()

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        blocksize=FRAME_SAMPLES,
        dtype="float32",
        callback=audio_callback,
    ):
        while active["running"]:
            time.sleep(0.05)

    vad_thread.join(timeout=2.0)
    tr_thread.join(timeout=2.0)


async def question_detector():
    buffer = ""
    while True:
        try:
            timeout = UTTERANCE_DEBOUNCE_S if buffer else None
            line = await asyncio.wait_for(speech_q.get(), timeout=timeout)
            print("[FRAGMENT]", line)
            buffer = (buffer + " " + line).strip() if buffer else line
        except asyncio.TimeoutError:
            if buffer:
                await questions_q.put(buffer)
                buffer = ""


def _trim_history():
    if len(conversation_history) > MAX_HISTORY_MESSAGES:
        del conversation_history[: len(conversation_history) - MAX_HISTORY_MESSAGES]


# ----- Streaming response pipeline -----

_SENTENCE_END = re.compile(r"([.!?])(\s|$)")


def _try_split_sentence(buf: str) -> tuple[str | None, str]:
    m = _SENTENCE_END.search(buf)
    if not m:
        return None, buf
    end = m.end()
    return buf[:end].strip(), buf[end:]


async def stream_gpt(question: str, resp: ResponseSession):
    try:
        stream = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": build_system_prompt()},
                *conversation_history,
            ],
            max_tokens=50,
            stream=True,
        )
        buf = ""
        async for chunk in stream:
            if resp.cancel_event.is_set():
                break
            delta = chunk.choices[0].delta.content or ""
            if not delta:
                continue
            buf += delta
            resp.full_answer += delta
            while True:
                sentence, buf = _try_split_sentence(buf)
                if not sentence:
                    break
                await resp.sentence_q.put(sentence)
        if buf.strip() and not resp.cancel_event.is_set():
            await resp.sentence_q.put(buf.strip())
    except Exception as e:
        print(f"OpenAI error: {e}")
        if not resp.full_answer:
            resp.full_answer = "I don't know."
            await resp.sentence_q.put("I don't know.")
    finally:
        await resp.sentence_q.put(None)


async def synth_worker(resp: ResponseSession):
    try:
        while True:
            sentence = await resp.sentence_q.get()
            if sentence is None or resp.cancel_event.is_set():
                return
            audio = await asyncio.to_thread(_piper_synthesize, sentence)
            if resp.cancel_event.is_set():
                return
            await resp.audio_q.put(audio)
    finally:
        await resp.audio_q.put(None)


async def play_worker(resp: ResponseSession):
    started = False
    try:
        while True:
            audio = await resp.audio_q.get()
            if audio is None or resp.cancel_event.is_set():
                break
            if not started:
                _start_playing()
                started = True
            try:
                await asyncio.to_thread(sd.play, audio, TTS_SAMPLE_RATE, blocking=True)
            except Exception as e:
                print(f"Play error: {e}")
                break
            if resp.cancel_event.is_set():
                break
        if started and not resp.cancel_event.is_set():
            await asyncio.sleep(0.2)  # let room reverb die before mic re-engages
    finally:
        if started:
            _stop_playing()


async def question_handler():
    global active_response, _last_assistant_reply
    while True:
        question = await questions_q.get()
        print(f"\n[QUESTION] {question}")

        conversation_history.append({"role": "user", "content": question})
        _trim_history()
        append_history({"role": "user", "content": question})

        resp = ResponseSession()
        active_response = resp

        await asyncio.gather(
            stream_gpt(question, resp),
            synth_worker(resp),
            play_worker(resp),
        )

        active_response = None

        final = resp.full_answer.strip()
        if final:
            conversation_history.append({"role": "assistant", "content": final})
            _trim_history()
            append_history({"role": "assistant", "content": final})
            _last_assistant_reply = final

        if resp.cancel_event.is_set():
            print("[INTERRUPTED]")
        else:
            print(f"[ANSWER] {final}")


# ----- TTS helpers -----

def _clean_tts_audio(audio: np.ndarray, sr: int) -> np.ndarray:
    fade_in = int(sr * 0.01)
    fade_out = int(sr * 0.04)
    if fade_in > 0 and len(audio) > fade_in:
        audio[:fade_in] *= np.linspace(0, 1, fade_in, dtype=np.float32)
    if fade_out > 0 and len(audio) > fade_out:
        audio[-fade_out:] *= np.linspace(1, 0, fade_out, dtype=np.float32)
    return np.clip(audio, -1.0, 1.0)


def _piper_synthesize(text: str) -> np.ndarray:
    chunks = [chunk.audio_float_array for chunk in tts_voice.synthesize(text)]
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    pcm = np.concatenate(chunks).astype(np.float32)
    return _clean_tts_audio(pcm, TTS_SAMPLE_RATE)


# ----- Entrypoint -----

async def main_async():
    loop = asyncio.get_running_loop()
    mic_holder: dict = {}

    t = Thread(target=mic_thread_fn, args=(loop, mic_holder), daemon=True)
    t.start()

    stop_event = asyncio.Event()

    def _on_stop(*_):
        stop_event.set()

    loop.add_signal_handler(signal.SIGINT, _on_stop)
    loop.add_signal_handler(signal.SIGTERM, _on_stop)

    print(
        f"Ready. Barge-in: {'ON' if BARGE_IN else 'OFF'}. "
        f"Profile: {profile.get('name') or 'anonymous'}. "
        f"History: {len(conversation_history)} prior messages."
    )

    tasks = [
        asyncio.create_task(question_detector(), name="question_detector"),
        asyncio.create_task(question_handler(), name="question_handler"),
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
