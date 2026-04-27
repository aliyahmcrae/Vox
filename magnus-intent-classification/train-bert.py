import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

# =========================
# 1. LOAD LABEL MAPPING
# =========================

with open("intent_names.txt") as f:
    intent_names = sorted(i.strip() for i in set(f))

id2label = {i: name for i, name in enumerate(intent_names)}
label2id = {v: k for k, v in id2label.items()}

# =========================
# 2. LOAD DATASET
# =========================

dataset = load_dataset("DeepPavlov/hwu64")

train_data = dataset["train"]

texts = train_data["text"] if "text" in train_data.column_names else train_data["utterance"]
labels = train_data["label"]

# Convert to HF dataset format
from datasets import Dataset

hf_dataset = Dataset.from_dict({
    "text": texts,
    "label": labels
})

# Train/test split
hf_dataset = hf_dataset.train_test_split(test_size=0.1)

# =========================
# 3. TOKENIZATION
# =========================

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=64
    )

tokenized = hf_dataset.map(tokenize, batched=True)

# rename label column → required by Trainer
tokenized = tokenized.rename_column("label", "labels")
tokenized.set_format("torch")

# =========================
# 4. LOAD MODEL
# =========================

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(id2label),
    id2label=id2label,
    label2id=label2id
)

# =========================
# 5. TRAINING CONFIG
# =========================

training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50,
)

# =========================
# 6. METRICS
# =========================

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    accuracy = (preds == labels).mean()
    return {"accuracy": accuracy}

# =========================
# 7. TRAINER
# =========================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    compute_metrics=compute_metrics
)

# =========================
# 8. TRAIN
# =========================

def train():
    trainer.train()
    trainer.save_model("./4_27_intent_bert")

# =========================
# 9. INFERENCE
# =========================

def classify(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]

    scores = {
        id2label[i]: float(probs[i])
        for i in range(len(id2label))
    }

    return dict(sorted(scores.items(), key=lambda x: -x[1]))

# =========================
# 10. QUICK TEST
# =========================

if __name__ == "__main__":
    train()

    examples = [
        "set an alarm for 7am tomorrow",
        "what's the weather like?",
        "remind me to call mom",
        "play music",
        "I forgot my password"
    ]

    for e in examples:
        print(e)
        print(classify(e))
        print("=" * 40)