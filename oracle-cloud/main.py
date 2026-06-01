from piper.voice import PiperVoice
from openai import AsyncOpenAI
from typing import Callable, Awaitable, Any
import asyncio
import re
import numpy as np
import json
import base64
import tomllib
import websockets


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


def piper_synthesize(text: str, tts_voice, sample_rate) -> np.ndarray:
    print(
        f"[piper_synthesize] synthesizing text_len={len(text)} sample_rate={sample_rate}")
    chunks = [chunk.audio_float_array for chunk in tts_voice.synthesize(text)]
    if not chunks:
        print("[piper_synthesize] no chunks produced, returning empty array")
        return np.zeros(0, dtype=np.float32)
    audio = np.concatenate(chunks).astype(np.float32)
    print(f"[piper_synthesize] concatenated audio samples={len(audio)}")

    fade_in = int(sample_rate * 0.01)
    fade_out = int(sample_rate * 0.04)
    if fade_in > 0 and len(audio) > fade_in:
        audio[:fade_in] *= np.linspace(0, 1, fade_in, dtype=np.float32)
    if fade_out > 0 and len(audio) > fade_out:
        audio[-fade_out:] *= np.linspace(1, 0, fade_out, dtype=np.float32)
    out = np.clip(audio, -1.0, 1.0)
    print(f"[piper_synthesize] returning audio length={len(out)}")
    return out


class ResponsePipeline:
    prompt_queue: asyncio.Queue
    sentence_queue: asyncio.Queue
    audio_out_callback: Callable[[np.ndarray], Awaitable[None]]
    tts_model: Any
    openai: Any
    conversation: Any

    MODEL: str
    MAX_TOKENS: int
    SAMPLE_RATE: int

    def __init__(self, config: dict[str, Any], tts_model, openai):
        self.prompt_queue = asyncio.Queue()
        self.sentence_queue = asyncio.Queue()
        self.tts_model = tts_model
        self.audio_out_callback = None
        self.openai = openai
        self.conversation = None

        self.MODEL = config["MODEL"]
        self.MAX_TOKENS = config["MAX_TOKENS"]
        self.SAMPLE_RATE = config["SAMPLE_RATE"]

    def set_callback(self, callback):
        self.audio_out_callback = callback

    async def submit_prompt(self, text: str):
        print(f"[ResponsePipeline] submit_prompt: queuing text={text!r}")
        await self.prompt_queue.put(text)

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

            # synthesize on a thread to avoid blocking the event loop
            print(
                "[ResponsePipeline] generate_audio: invoking piper_synthesize on thread")
            audio = await asyncio.to_thread(
                piper_synthesize,
                sentence,
                self.tts_model,
                self.SAMPLE_RATE
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
                if data.get("type") == "prompt":
                    if self.handle_prompt:
                        print(
                            f"[Relay] handle_pi: scheduling handle_prompt with data={data.get('data')!r}")
                        # Pi uses "data" field for the text prompt
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


async def main():
    # load secrets and config
    with open("cache/secrets.json") as f:
        secrets = json.load(f)
    with open("config.toml", "rb") as t:
        conf = tomllib.load(t)

    OPENAI_KEY = secrets.get("openai")
    client = AsyncOpenAI(api_key=OPENAI_KEY)
    tts = PiperVoice.load("./cache/en_US-lessac-medium.onnx")

    # conf["gpt"] contains MODEL / MAX_TOKENS / SAMPLE_RATE
    response_pipeline = ResponsePipeline(conf["gpt"], tts, client)
    relay = Relay()

    relay.set_callback(response_pipeline.submit_prompt)
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
    asyncio.run(main())
