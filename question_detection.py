from transformers import pipeline

# Small-ish instruction model (you can swap this)
pipe = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    max_new_tokens=50
)

def extract_last_question(text):
    prompt = f"""
You are an information extraction system.

Task:
- Find the LAST question in the text
- If no question exists, return: NONE
- Only output the question, nothing else

Text:
{text}

Answer:
"""

    result = pipe(prompt)[0]["generated_text"]

    # crude cleanup (because tiny models ramble)
    answer = result.split("Answer:")[-1].strip()

    return answer


if __name__ == "__main__":
    sample = """
    Hey, are you coming to the meeting later?
    I think we should review the budget first.
    Also, did you finish the report?
    Anyway, let's talk tomorrow.
    """

    print(extract_last_question(sample))
