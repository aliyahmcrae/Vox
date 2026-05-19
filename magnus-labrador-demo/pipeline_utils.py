from embeddings import classify as embed_classify
from use_bert import classify as bert_classify
# from decision_layer import decide_from_embeddings

def normalize_embeddings(text):
    scores = embed_classify(text)

    top_intent = max(scores, key=scores.get)
    confidence = scores[top_intent]

    print("[EMBED TOP]", top_intent, confidence)

    if confidence > 0.75:
        return "RESPOND"

    elif confidence < 0.45:
        return "IGNORE"

    else:
        return "UNCERTAIN"

def normalize_bert(text):
    scores = bert_classify(text)

    top_intent = max(scores, key=scores.get)
    confidence = scores[top_intent]

    print("[BERT TOP]", top_intent, confidence)

    if confidence > 0.72:
        return "RESPOND"

    elif confidence < 0.45:
        return "IGNORE"

    else:
        return "UNCERTAIN"

def final_pipeline(text):
    emb_decision = normalize_embeddings(text)

    # -------------------------
    # STRONG YES → RESPOND
    # -------------------------
    if emb_decision == "RESPOND":
        return "RESPOND"

    # -------------------------
    # STRONG NO → IGNORE
    # -------------------------
    if emb_decision == "IGNORE":
        # ⚠️ NEW: don't trust embeddings fully
        bert_decision = normalize_bert(text)

        if bert_decision == "RESPOND":
            return "RESPOND"

        return "IGNORE"

    # -------------------------
    # UNCERTAIN → USE BERT
    # -------------------------
    return normalize_bert(text)