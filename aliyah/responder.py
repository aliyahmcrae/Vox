import os

USE_LOCAL = False
LOCAL_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
OPENAI_MODEL = "gpt-3.5-turbo"

SYSTEM_PROMPT = (
    "You are Vox, a concise and helpful voice assistant. "
    "Keep responses short — 1-3 sentences max. No markdown."
)

_local_pipeline = None


def _load_local():
    global _local_pipeline
    if _local_pipeline is None:
        from transformers import pipeline
        _local_pipeline = pipeline(
            "text-generation",
            model=LOCAL_MODEL,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7,
        )
    return _local_pipeline


def respond(text: str, history: list[dict] = None) -> str:
    if USE_LOCAL:
        return _respond_local(text, history or [])
    else:
        return _respond_openai(text, history or [])


def _respond_openai(text: str, history: list[dict]) -> str:
    import openai
    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += history
    messages.append({"role": "user", "content": text})
    response = client.chat.completions.create(model=OPENAI_MODEL, messages=messages)
    return response.choices[0].message.content.strip()


def _respond_local(text: str, history: list[dict]) -> str:
    pipe = _load_local()
    prompt = f"<|system|>{SYSTEM_PROMPT}</s>\n"
    for turn in history[-4:]:
        prompt += f"<|{turn['role']}|>{turn['content']}</s>\n"
    prompt += f"<|user|>{text}</s>\n<|assistant|>"
    result = pipe(prompt)[0]["generated_text"]
    return result[len(prompt):].strip()


if __name__ == "__main__":
    q = "What time is it?"
    print(f"Q: {q}\nA: {respond(q)}")
