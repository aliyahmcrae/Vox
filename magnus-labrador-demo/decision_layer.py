import numpy as np

# tuneable thresholds
RESPOND_THRESHOLD = 0.25
IGNORE_THRESHOLD = 0.10
GAP_THRESHOLD = 0.02  # difference between top 1 and top 2

def decide_from_embeddings(score_dict):

    sorted_items = sorted(score_dict.items(), key=lambda x: -x[1])
    top_intent, top1 = sorted_items[0]
    top2 = sorted_items[1][1] if len(sorted_items) > 1 else 0

    gap = top1 - top2

    ALLOWED_INTENTS = [
        "weather_query",
        "qa_factoid",
        "calendar_set",
        "play_music",
        "alarm_set",
        "transport_ticket",
        "takeaway_order",
        "calendar_query",
        "datetime_query",
        "iot_hue_lightoff",
        "iot_hue_lighton",
        "general_explain",
    ]

    # 🚫 FIRST: filter intents
    if top_intent not in ALLOWED_INTENTS:
        return "IGNORE"

    # ✅ THEN: apply confidence rules
    if top1 >= RESPOND_THRESHOLD and gap >= GAP_THRESHOLD:
        return "RESPOND"

    if top1 <= IGNORE_THRESHOLD:
        return "IGNORE"

    return "UNCERTAIN"