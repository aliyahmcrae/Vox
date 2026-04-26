from transformers import pipeline

USE_FINETUNED = False
FINETUNED_MODEL_PATH = "./models/intent-classifier"

INTENT_LABELS = ["question", "command", "statement", "greeting", "unknown"]
CONFIDENCE_THRESHOLD = 0.45

if USE_FINETUNED:
    _classifier = pipeline("text-classification", model=FINETUNED_MODEL_PATH)
else:
    _classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")


def classify(text: str) -> dict:
    if not text or not text.strip():
        return {"intent": "unknown", "confidence": 0.0, "should_respond": False}

    if USE_FINETUNED:
        result = _classifier(text)[0]
        intent = result["label"].lower()
        confidence = result["score"]
    else:
        result = _classifier(text, candidate_labels=INTENT_LABELS)
        intent = result["labels"][0]
        confidence = result["scores"][0]

    if confidence < CONFIDENCE_THRESHOLD:
        intent = "unknown"

    should_respond = intent in ("question", "command", "greeting")

    return {
        "intent": intent,
        "confidence": round(confidence, 3),
        "should_respond": should_respond,
    }


if __name__ == "__main__":
    tests = [
        "What's the weather like today?",
        "Play some music",
        "I was just telling John about you",
        "Hey there",
        "The meeting is at 3pm",
        "Can you set a timer for 10 minutes",
    ]
    for t in tests:
        print(f"'{t}'\n  → {classify(t)}\n")
