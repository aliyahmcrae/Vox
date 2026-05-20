from piper.voice import PiperVoice
from openai import AsyncOpenAI
from typing import Callable, Awaitable, Any
import asyncio
import re
import numpy as np


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
    chunks = [chunk.audio_float_array for chunk in tts_voice.synthesize(text)]
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    audio = np.concatenate(chunks).astype(np.float32)

    fade_in = int(sample_rate * 0.01)
    fade_out = int(sample_rate * 0.04)
    if fade_in > 0 and len(audio) > fade_in:
        audio[:fade_in] *= np.linspace(0, 1, fade_in, dtype=np.float32)
    if fade_out > 0 and len(audio) > fade_out:
        audio[-fade_out:] *= np.linspace(1, 0, fade_out, dtype=np.float32)
    return np.clip(audio, -1.0, 1.0)


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
        await self.prompt_queue.put(text)

    async def generate_responses(self):
        while True:
            prompt = [{"role": "user", "content": await self.prompt_queue.get()}]

            if self.conversation is None:
                self.conversation = await self.openai.conversations.create()
                prompt = [{"role": "system", "content": BASE_PROMPT}] + prompt

            stream = await self.openai.chat.responses.create(
                model=self.MODEL,
                input=prompt,
                max_tokens=self.MAX_TOKENS,
                stream=True,
                conversation=self.conversation.id
            )

            buf = ""

            async for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                if not delta:
                    continue
                buf += delta
                while True:
                    sentence, buf = try_split_sentence(buf)
                    if not sentence:
                        break
                    await self.sentence_queue.put(sentence)
            if buf.strip():
                await self.sentence_queue.put(buf.strip())

    async def generate_audio(self):
        while True:
            sentence = await self.sentence_queue.get()

            audio = await asyncio.to_thread(
                piper_synthesize,
                sentence,
                tts_model,
                self.SAMPLE_RATE
            )

            if self.audio_out_callback:
                await self.audio_out_callback(audio)

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
        self.handle_prompt = handle_prompt

    async def handle_pi(self, ws):
        self.pi_socket = ws
        print("pi connected")

        try:
            async for msg in ws:
                data = json.loads(msg)

                if data.get("type") == "prompt":
                    if self.handle_prompt:
                        asyncio.create_task(
                            self.handle_prompt(data["text"])
                        )

        finally:
            self.pi_socket = None
            print("pi disconnected")

    def handle_client(self, ws):
        self.client_socket = ws
        print("client connected")

    async def msg_client(self, data):
        if self.client_socket:
            self.client_socket.send(data)

    async def router(self, ws):
        path = ws.request.path

        if path == "/ws/pi":
            await self.handle_pi(ws)

        elif path == "/ws/client":
            self.handle_client(ws)

        else:
            await ws.close()


async def main():
    with open("secrets.json") as f:
        OPENAI_KEY = json.load(f)["openai"]

    client = AsyncOpenAI(api_key=OPENAI_KEY)
    tts = PiperVoice.load("./en_US-lessac-medium.onnx")

    response_pipeline = ResponsePipeline(config, tts, client)
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


if __name__ == "__main__":
    asyncio.run(main())
