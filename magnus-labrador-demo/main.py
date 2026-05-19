import asyncio
import json
import random
import signal
import soundfile
import sys
import time
from threading import Thread
from pathlib import Path


from openai import AsyncOpenAI
from openai.helpers import LocalAudioPlayer

import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel

from pipeline_utils import final_pipeline, normalize_bert, normalize_embeddings
from embeddings import classify
from logger_utils import log_interaction


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

is_speaking = False

def whisper_mic_thread(loop):
    model = WhisperModel("base")

    samplerate = 16000
    chunk_duration = 2.5

    print("[WHISPER] Mic started...")

    while True:
        # DO NOT LISTEN WHILE SPEAKING
        if is_speaking:
            time.sleep(0.1)
            continue

        print("[WHISPER] Listening...")

        audio = sd.rec(
            int(chunk_duration * samplerate),
            samplerate=samplerate,
            channels=1,
            dtype="float32"
        )
        sd.wait()

        audio = np.squeeze(audio)

        segments, _ = model.transcribe(audio)
        text = " ".join([seg.text.strip() for seg in segments]).strip()

        if text:
            print("[WHISPER TEXT]", text)
            asyncio.run_coroutine_threadsafe(speech_q.put(text), loop)


async def question_detector():
    buffer = ""
    last_update = asyncio.get_event_loop().time()

    while True:
        line = await speech_q.get()
        now = asyncio.get_event_loop().time()

        print("[RAW LINE]", line)

        # -----------------------------
        # ACCUMULATE SPEECH
        # -----------------------------
        if buffer:
            buffer += " " + line
        else:
            buffer = line

        last_update = now

        # -----------------------------
        # WAIT FOR USER TO FINISH TALKING
        # -----------------------------
        await asyncio.sleep(0.6)

        # If no new speech came in recently → finalize
        if asyncio.get_event_loop().time() - last_update > 0.5:
            print("\n[FINAL TEXT]", buffer)

            decision = final_pipeline(buffer)

            print(f"[DECISION] {decision}")
            print(f"[BERT] {normalize_bert(buffer)}")

            try:
                scores = classify(buffer)

                print("[TOP INTENTS]", list(classify(buffer).items())[:3])

                top_intent = max(scores, key=scores.get)
                top_score = scores[top_intent]

                embed_decision = normalize_embeddings(buffer)
                bert_decision = normalize_bert(buffer)

                log_interaction(
                    text=buffer,
                    embed_decision=embed_decision,
                    bert_decision=bert_decision,
                    final_decision=decision,
                    top_intent=top_intent,
                    top_score=top_score,
                )
            except Exception as e:
                print("[LOGGING ERROR]", e)
 
            # -----------------------------
            # TRIGGER RESPONSE (SIMPLIFIED)
            # -----------------------------
            if decision == "RESPOND":
                print("[TRIGGER] Sending to LLM:", buffer)
                await questions_q.put(buffer)

            elif decision == "IGNORE":
                print("[IGNORE] Dropping buffer")

            elif decision == "UNCERTAIN":
                print("[UNCERTAIN] Skipping")

            # reset buffer after decision
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
    global is_speaking

    while True:
        answer = await answers_q.get()

        try:
            async with client.audio.speech.with_streaming_response.create(
                model="gpt-4o-mini-tts",
                voice="alloy",
                input=answer,
                response_format="wav",
            ) as response:

                is_speaking = True

                print("Playing audio!")
                await LocalAudioPlayer().play(response)

        except Exception as e:
            print("TTS error:", e)

        finally:
            is_speaking = False
            await asyncio.sleep(0.3)


async def main_async():
    loop = asyncio.get_running_loop()

    # mic holder shared between threads
    mic_holder = {}

    # start mic thread
    t = Thread(target=whisper_mic_thread, args=(loop,), daemon=True)
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
