from __future__ import annotations

from openai import AsyncOpenAI
from typing import Callable, Awaitable, Any
import asyncio
import os
import re
import sys
import numpy as np
import json
import base64
import tomllib
from collections import deque

# `kokoro` (TTS) and `websockets` (relay server) are imported lazily inside
# main() so the --text smoke-test mode can run with only `openai` installed.
if False:  # type-checking only
    from kokoro import KPipeline


INTENT_PROMPT = """You are the turn-detection component of a voice assistant named Vox.

You receive a rolling transcript of what a user has said. It comes from streaming speech-to-text, so it may be fragmented, mid-thought, or contain transcription errors.

Decide whether the user has finished a complete request or question that is directed at the assistant and is ready to be answered.

Set ready=true only when BOTH are true:
* The utterance forms a complete, self-contained request or question (a follow-up like "what about tomorrow?" counts as complete).
* The user appears to have finished the thought rather than trailing off mid-sentence.

Set ready=false when the user is still mid-sentence, thinking aloud, hesitating, or clearly talking to someone other than the assistant.

When ready=true, set query to the cleaned-up request as a single clear sentence: fix obvious transcription artifacts and drop filler words, but do not add information. When ready=false, set query to an empty string.
"""


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

SENTENCE_END = re.compile(r"([.!?])(\s|$)")


def try_split_sentence(buf: str) -> tuple[str | None, str]:
    m = SENTENCE_END.search(buf)
    if not m:
        return None, buf
    end = m.end()
    return buf[:end].strip(), buf[end:]


TTS_VOICE = "af_heart"
KOKORO_LANG = "a"  # 'a' = American English; 'b' = British English
KOKORO_RATE = 24000  # Kokoro outputs at 24kHz mono float32


def kokoro_synthesize(text: str, pipeline: KPipeline, voice: str, sample_rate: int) -> np.ndarray:
    print(
        f"[kokoro_synthesize] synthesizing text_len={len(text)} voice={voice}")
    chunks: list[np.ndarray] = []
    for _gs, _ps, audio in pipeline(text, voice=voice, speed=1):
        if hasattr(audio, "numpy"):
            audio = audio.numpy()
        chunks.append(np.asarray(audio, dtype=np.float32))

    if not chunks:
        print("[kokoro_synthesize] no chunks produced, returning empty array")
        return np.zeros(0, dtype=np.float32)
    audio = np.concatenate(chunks)
    print(
        f"[kokoro_synthesize] received {len(audio)} samples @ {KOKORO_RATE} Hz")

    # Browser hardcodes 16kHz playback; downsample so pitch/speed stay correct.
    if sample_rate != KOKORO_RATE and len(audio) > 0:
        duration = len(audio) / KOKORO_RATE
        new_len = int(duration * sample_rate)
        if new_len > 0:
            src_t = np.linspace(0, duration, len(audio), endpoint=False)
            dst_t = np.linspace(0, duration, new_len, endpoint=False)
            audio = np.interp(dst_t, src_t, audio).astype(np.float32)

    fade_in = int(sample_rate * 0.01)
    fade_out = int(sample_rate * 0.04)
    if fade_in > 0 and len(audio) > fade_in:
        audio[:fade_in] *= np.linspace(0, 1, fade_in, dtype=np.float32)
    if fade_out > 0 and len(audio) > fade_out:
        audio[-fade_out:] *= np.linspace(1, 0, fade_out, dtype=np.float32)
    out = np.clip(audio, -1.0, 1.0)
    print(
        f"[kokoro_synthesize] returning audio length={len(out)} @ {sample_rate} Hz")
    return out


class ResponsePipeline:
    transcript_queue: asyncio.Queue
    prompt_queue: asyncio.Queue
    sentence_queue: asyncio.Queue
    audio_out_callback: Callable[[np.ndarray], Awaitable[None]]
    openai: Any
    tts_pipeline: KPipeline
    conversation: Any

    MODEL: str
    MAX_TOKENS: int
    SAMPLE_RATE: int
    INTENT_MODEL: str
    CONTEXT_LENGTH: int

    def __init__(self, config: dict[str, Any], intent_config: dict[str, Any], openai, tts_pipeline: KPipeline):
        self.transcript_queue = asyncio.Queue()
        self.prompt_queue = asyncio.Queue()
        self.sentence_queue = asyncio.Queue()
        self.audio_out_callback = None
        self.openai = openai
        self.tts_pipeline = tts_pipeline
        self.conversation = None

        self.MODEL = config["MODEL"]
        self.MAX_TOKENS = config["MAX_TOKENS"]
        self.SAMPLE_RATE = config["SAMPLE_RATE"]
        self.INTENT_MODEL = intent_config["MODEL"]
        self.CONTEXT_LENGTH = intent_config["CONTEXT_LENGTH"]

    def set_callback(self, callback):
        self.audio_out_callback = callback

    async def submit_transcript(self, text: str):
        print(f"[ResponsePipeline] submit_transcript: queuing text={text!r}")
        await self.transcript_queue.put(text)

    async def detect_intent(self):
        # Buffer recent transcript lines and let the model decide when a
        # complete, assistant-directed request has been spoken.
        context = deque(maxlen=self.CONTEXT_LENGTH)
        print("[ResponsePipeline] detect_intent: started")
        while True:
            line = await self.transcript_queue.get()
            context.append(line)
            transcript = " ".join(context).strip()
            if not transcript:
                continue

            print(f"[ResponsePipeline] detect_intent: classifying transcript={transcript!r}")
            resp = await self.openai.responses.create(
                model=self.INTENT_MODEL,
                input=[
                    {"role": "system", "content": INTENT_PROMPT},
                    {"role": "user", "content": transcript},
                ],
                text={"format": {
                    "type": "json_schema",
                    "name": "intent",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "ready": {"type": "boolean"},
                            "query": {"type": "string"},
                        },
                        "required": ["ready", "query"],
                        "additionalProperties": False,
                    },
                }},
            )

            try:
                result = json.loads(resp.output_text)
            except (json.JSONDecodeError, AttributeError) as e:
                print(f"[ResponsePipeline] detect_intent: failed to parse result: {e}")
                continue

            if result.get("ready"):
                query = result.get("query") or transcript
                print(f"[ResponsePipeline] detect_intent: ready, dispatching query={query!r}")
                await self.prompt_queue.put(query)
                context.clear()
            else:
                print(f"[ResponsePipeline] detect_intent: not ready, buffering")

    async def generate_responses(self):
        while True:
            prompt = [{"role": "user", "content": await self.prompt_queue.get()}]
            print(f"[ResponsePipeline] generate_responses: got prompt")

            if self.conversation is None:
                self.conversation = await self.openai.conversations.create()
                prompt = [{"role": "system", "content": BASE_PROMPT}] + prompt

            stream = await self.openai.responses.create(
                model=self.MODEL,
                input=prompt,
                stream=True,
                conversation=self.conversation.id,
                max_output_tokens=self.MAX_TOKENS,
            )

            buf = ""

            async for event in stream:
                if event.type != "response.output_text.delta":
                    continue

                delta = event.delta
                buf += delta
                print(f"Got delta:", delta)

                while True:
                    sentence, buf = try_split_sentence(buf)
                    if not sentence:
                        break

                    await self.sentence_queue.put(sentence)

            if buf.strip():
                print(f"Finished sentence: {buf.strip()}")
                await self.sentence_queue.put(buf.strip())

    async def generate_audio(self):
        while True:
            sentence = await self.sentence_queue.get()
            print(
                f"[ResponsePipeline] generate_audio: got sentence={sentence!r}")

            print(
                "[ResponsePipeline] generate_audio: invoking kokoro_synthesize on thread")
            audio = await asyncio.to_thread(
                kokoro_synthesize,
                sentence,
                self.tts_pipeline,
                TTS_VOICE,
                self.SAMPLE_RATE,
            )
            print(
                f"[ResponsePipeline] generate_audio: synthesized audio length={getattr(audio, 'size', 'n/a')}")

            if self.audio_out_callback:
                print(
                    "[ResponsePipeline] generate_audio: invoking audio_out_callback")
                await self.audio_out_callback(audio)
                print(
                    "[ResponsePipeline] generate_audio: audio_out_callback completed")

    async def run(self):
        async with asyncio.TaskGroup() as tg:
            tg.create_task(self.detect_intent())
            tg.create_task(self.generate_responses())
            tg.create_task(self.generate_audio())


class Relay:
    def __init__(self):
        self.pi_socket = None
        self.client_socket = None
        self.handle_prompt = None

    def set_callback(self, callback):
        self.handle_prompt = callback

    async def handle_pi(self, ws):
        self.pi_socket = ws
        print("pi connected")

        try:
            async for msg in ws:
                print(
                    f"[Relay] handle_pi: received message type={type(msg)} len={len(msg) if hasattr(msg, '__len__') else 'n/a'}")
                if isinstance(msg, (bytes, bytearray)):
                    # pi should not be sending binary frames in this design
                    print(
                        "[Relay] handle_pi: unexpected binary frame from pi, ignoring")
                    continue
                try:
                    data = json.loads(msg)
                except Exception as e:
                    print(
                        f"[Relay] handle_pi: failed to parse json: {e} msg={msg!r}")
                    continue

                print(f"[Relay] handle_pi: parsed data={data}")
                if data.get("type") == "transcript":
                    if self.handle_prompt:
                        print(
                            f"[Relay] handle_pi: scheduling handle_prompt with data={data.get('data')!r}")
                        # Pi streams each completed transcript line in "data"
                        asyncio.create_task(
                            self.handle_prompt(data.get("data"))
                        )

        finally:
            self.pi_socket = None
            print("pi disconnected")

    async def handle_client(self, ws):
        self.client_socket = ws
        print("client connected")

        try:
            async for msg in ws:
                print(
                    f"[Relay] handle_client: received message type={type(msg)} len={len(msg) if hasattr(msg, '__len__') else 'n/a'}")
                # Binary frames from the browser contain Int16 PCM; forward them directly to the Pi
                if isinstance(msg, (bytes, bytearray)):
                    if self.pi_socket:
                        print(
                            f"[Relay] handle_client: forwarding binary audio to pi size={len(msg)}")
                        await self.pi_socket.send(msg)
                    else:
                        print(
                            "[Relay] handle_client: no pi connected; dropping binary frame")
                    continue

                # Text frames are control messages (register/start/stop, etc)
                try:
                    data = json.loads(msg)
                except Exception as e:
                    print(
                        f"[Relay] handle_client: failed to parse json: {e} msg={msg!r}")
                    continue

                print(f"[Relay] handle_client: parsed data={data}")
                if data.get("type") == "register":
                    print(
                        "[Relay] handle_client: register received, replying registered")
                    await ws.send(json.dumps({"type": "registered"}))
                elif data.get("type") == "start":
                    print("client started streaming")
                elif data.get("type") == "stop":
                    print("client stopped streaming")

        finally:
            self.client_socket = None
            print("client disconnected")

    async def msg_client(self, audio):
        if self.client_socket:
            print(
                f"[Relay] msg_client: preparing to send audio to client type={type(audio)}")
            # audio is expected to be a numpy float32 array in [-1,1]
            try:
                pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
                b = pcm.tobytes()
                print(
                    f"[Relay] msg_client: converted float audio to pcm bytes len={len(b)}")
            except Exception:
                # fallback: if already bytes, use as-is
                if isinstance(audio, (bytes, bytearray)):
                    b = bytes(audio)
                    print(
                        f"[Relay] msg_client: audio already bytes len={len(b)}")
                else:
                    print("[Relay] msg_client: unsupported audio type, dropping")
                    return
            b64 = base64.b64encode(b).decode("ascii")
            print(f"[Relay] msg_client: sending tts message size={len(b64)}")
            await self.client_socket.send(json.dumps({"type": "tts", "data": b64}))

    async def router(self, ws):
        path = ws.request.path
        print(f"[Relay] router: connection path={path}")

        if path == "/cse481/ws/pi":
            print("[Relay] router: routing to handle_pi")
            await self.handle_pi(ws)

        elif path == "/cse481/ws/client":
            print("[Relay] router: routing to handle_client")
            await self.handle_client(ws)

        else:
            print(f"[Relay] router: unknown path {path}, closing")
            await ws.close()


def load_openai_key() -> str:
    # Prefer the shared secrets file; fall back to the env var for local runs.
    try:
        with open("cache/secrets.json") as f:
            key = json.load(f).get("openai")
            if key:
                return key
    except FileNotFoundError:
        pass
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit(
            "No OpenAI key found: provide cache/secrets.json or set OPENAI_API_KEY.")
    return key


async def run_text_mode():
    """Feed typed transcript lines through the intent gate + answer model and
    print the reply, skipping the relay server and TTS. Lets you sanity-check
    the gate without the Pi or browser. Each typed line == one Pi chunk."""
    with open("config.toml", "rb") as t:
        conf = tomllib.load(t)

    client = AsyncOpenAI(api_key=load_openai_key())
    pipeline = ResponsePipeline(conf["gpt"], conf["intent"], client, None)

    async def printer():
        while True:
            sentence = await pipeline.sentence_queue.get()
            print(f"\nVOX> {sentence}", flush=True)

    print("Text smoke-test mode. Each line you type is treated like one Pi "
          "transcript chunk.")
    print("The gate decides when a complete request has been spoken. "
          "Ctrl-D to exit.\n")

    loop = asyncio.get_running_loop()
    tasks = [
        asyncio.create_task(pipeline.detect_intent()),
        asyncio.create_task(pipeline.generate_responses()),
        asyncio.create_task(printer()),
    ]
    try:
        while True:
            line = await loop.run_in_executor(None, sys.stdin.readline)
            if not line:  # EOF (Ctrl-D)
                break
            line = line.strip()
            if line:
                await pipeline.submit_transcript(line)
        await asyncio.sleep(2)  # let any in-flight reply finish printing
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    print("\nGoodbye!")


async def main():
    from kokoro import KPipeline
    import websockets

    # load config; key comes from secrets.json or OPENAI_API_KEY
    with open("config.toml", "rb") as t:
        conf = tomllib.load(t)

    client = AsyncOpenAI(api_key=load_openai_key())

    print(f"[main] loading Kokoro pipeline (lang={KOKORO_LANG})")
    tts_pipeline = KPipeline(lang_code=KOKORO_LANG)

    # conf["gpt"] has MODEL / MAX_TOKENS / SAMPLE_RATE; conf["intent"] has the
    # intent-classifier MODEL / CONTEXT_LENGTH
    response_pipeline = ResponsePipeline(
        conf["gpt"], conf["intent"], client, tts_pipeline)
    relay = Relay()

    relay.set_callback(response_pipeline.submit_transcript)
    response_pipeline.set_callback(relay.msg_client)

    async with websockets.serve(
        relay.router,
        "0.0.0.0",
        8765,
        max_size=None,
        ping_interval=20
    ), asyncio.TaskGroup() as tg:
        tg.create_task(response_pipeline.run())
        print("Started!")

    print("Goodbye!")

if __name__ == "__main__":
    if "--text" in sys.argv:
        asyncio.run(run_text_mode())
    else:
        asyncio.run(main())
