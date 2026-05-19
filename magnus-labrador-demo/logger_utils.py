import csv
import os

LOG_FILE = "interaction_log.csv"

def log_interaction(
    text,
    embed_decision,
    bert_decision,
    final_decision,
    top_intent,
    top_score,
):
    file_exists = os.path.exists(LOG_FILE)

    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)

        # write header once
        if not file_exists:
            writer.writerow([
                "text",
                "embed_decision",
                "bert_decision",
                "final_decision",
                "top_intent",
                "top_score",
            ])

        writer.writerow([
            text,
            embed_decision,
            bert_decision,
            final_decision,
            top_intent,
            top_score,
        ])