import asyncio
import json
import random
import signal
import soundfile
from threading import Thread
from pathlib import Path

from openai import AsyncOpenAI
from openai.helpers import LocalAudioPlayer

from moonshine_voice import (
    MicTranscriber,
    TranscriptEventListener,
    get_model_for_language,
)

# Load API key
with open("./labrador/secrets.json") as f:
    secrets = json.load(f)

client = AsyncOpenAI(api_key=secrets["openai"])

# Queues
speech_q = asyncio.Queue()
questions_q = asyncio.Queue()
answers_q = asyncio.Queue()
play_q = asyncio.Queue()

CACHE_DIR = Path("./labrador/cues")

# The assistant system prompt as required
ASSISTANT_PROMPT = """You are a voice assistant. Your responses must follow these rules strictly:

* Output plain text only. Do not use markdown, emojis, bullet points, or special formatting.
* Keep responses concise and natural for speech.
* Limit responses to what can be spoken in under 15 seconds (approximately 30–40 words).
* Prioritize clarity and directness over completeness.
* Do not include filler phrases, disclaimers, or unnecessary context.
* Answer the user’s question directly. If unsure, say you don’t know briefly.
* Avoid lists unless absolutely necessary, and keep them short and spoken naturally.
* Do not repeat the user’s question.
* Do not explain your reasoning unless explicitly asked.

Speak like a helpful human assistant: brief, clear, and to the point.
"""

# Listener that bridges moonshine event callbacks (executed in mic thread) -> asyncio queue
class Listener(TranscriptEventListener):
    def __init__(self, loop):
        super().__init__()
        self._loop = loop

    def on_line_started(self, event):
        print("\n[START]", event.line.text)

    def on_line_text_changed(self, event):
        print("[LIVE]", event.line.text, end="\r")

    def on_line_completed(self, event):
        text = event.line.text.strip()
        if not text:
            return
        # schedule putting into asyncio queue from this (mic) thread
        asyncio.run_coroutine_threadsafe(speech_q.put(text), self._loop)


def mic_thread_fn(loop, mic_holder):
    # This function runs in background thread; it creates and starts the MicTranscriber
    # and blocks on mic.start() until mic.stop() is called.
    model_path, model_arch = get_model_for_language("en")
    mic = MicTranscriber(model_path=model_path, model_arch=model_arch)
    listener = Listener(loop)
    mic.add_listener(listener)

    # expose mic to main thread for stopping
    mic_holder["mic"] = mic

    # This will block (per the demo) until mic.stop() is invoked.
    mic.start()


async def question_detector():
    # accumulate lines into a rolling buffer; when we see a question-mark we emit the buffered text
    buffer = ""
    while True:
        line = await speech_q.get()
        print("Detected line:", line)
        if buffer:
            buffer = buffer + " " + line
        else:
            buffer = line

        # Detect question mark anywhere in buffer; send the full buffer as question and reset buffer
        if "?" in buffer:
            q = buffer.strip()
            await questions_q.put(q)
            buffer = ""


async def play_random_wav():
    print("Playing...")
    # pick a random wav from tts_cache and play it (if any)
    if not CACHE_DIR.exists():
        return
    files = [p for p in CACHE_DIR.iterdir() if p.suffix.lower() == ".wav"]
    if not files:
        return
    chosen = random.choice(files)
    print("Chose file!", chosen)

    try:
        # soundfile.read is blocking; run it in a thread so we don't block the event loop
        data, sr = await asyncio.to_thread(soundfile.read, str(chosen), dtype="float32")
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        # pass samplerate to the player so playback is correct
        await LocalAudioPlayer().play(data)
    except Exception as e:
        print("Failed to play cue:", e)
        return


async def question_handler():
    while True:
        question = await questions_q.get()
        print("Generating response for question:", question)
        # Start the short prompt sound immediately
        play_task = asyncio.create_task(play_random_wav())
        await play_q.put(play_task)

        # Send to OpenAI with system prompt
        try:
            resp = await client.responses.create(
                model="gpt-5.2",
                input=[
                    {"role": "system", "content": ASSISTANT_PROMPT},
                    {"role": "user", "content": question},
                ],
            )
            # The library's response exposes output_text per the demo file
            answer = resp.output_text.strip() if hasattr(resp, "output_text") else ""
            if not answer:
                # Fallback: try to parse from generative output objects
                # We keep this defensive but prefer resp.output_text normally.
                try:
                    answer = "".join(m["content"]["text"] for m in resp.output if "content" in m)
                except Exception as e:
                    print(e)
                    answer = "I don't know."
        except Exception as e:
            print(e)
            answer = "I don't know."
        print("Response generated!")

        await answers_q.put(answer)


async def answer_player():
    while True:
        answer = await answers_q.get()
        # Use tts.speak which now will both synthesize (or use cache) and play
        try:
            print("Generating audio!")
            async with client.audio.speech.with_streaming_response.create(
                model="gpt-4o-mini-tts",
                voice="alloy",
                input=answer,
                response_format="wav",  # low latency playback
            ) as response:
                print("Waiting for cue playback tasks (if any)...")
                # Drain any queued play tasks and wait for each to finish before starting TTS playback.
                while True:
                    try:
                        play_task = play_q.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    try:
                        await play_task
                    except Exception as e:
                        print(e)

                print("Playing audio!")
                await LocalAudioPlayer().play(response)

        except Exception:
            # tolerate TTS failures gracefully
            pass


async def main_async():
    loop = asyncio.get_running_loop()

    # mic holder shared between threads
    mic_holder = {}

    # start mic thread
    t = Thread(target=mic_thread_fn, args=(loop, mic_holder), daemon=True)
    t.start()

    # register signal handlers to stop gracefully
    stop_event = asyncio.Event()

    def _on_stop(*_):
        stop_event.set()

    loop.add_signal_handler(signal.SIGINT, _on_stop)
    loop.add_signal_handler(signal.SIGTERM, _on_stop)

    # start background tasks
    tasks = [
        asyncio.create_task(question_detector(), name="question_detector"),
        asyncio.create_task(question_handler(), name="question_handler"),
        asyncio.create_task(answer_player(), name="answer_player"),
    ]

    # wait for stop event
    await stop_event.wait()

    # Begin shutdown: stop mic if available
    mic = mic_holder.get("mic")
    if mic is not None:
        try:
            mic.stop()
        except Exception:
            pass

    # give the mic thread a moment to exit
    t.join(timeout=2.0)

    # cancel tasks
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


if __name__ == "__main__":
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        pass
