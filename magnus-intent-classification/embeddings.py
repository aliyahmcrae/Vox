from datasets import load_dataset

with open("intent_names.txt") as f:
  intent_names = sorted(i.strip() for i in set(f))

id2label = {i: name for i, name in enumerate(intent_names)}

dataset = load_dataset("DeepPavlov/hwu64")

train_data = dataset["train"]

texts = train_data["text"] if "text" in train_data.column_names else train_data["utterance"]
labels = train_data["label"]

from collections import defaultdict

label_to_texts = defaultdict(list)

for text, label in zip(texts, labels):
    label_to_texts[label].append(text)

from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

# Embed all utterances
embeddings = model.encode(texts, normalize_embeddings=True)

label_to_vecs = {}

for label in label_to_texts:
    idxs = [i for i, l in enumerate(labels) if l == label]
    label_to_vecs[label] = embeddings[idxs]

from sklearn.metrics.pairwise import cosine_similarity

def classify(query):
    q_emb = model.encode([query], normalize_embeddings=True)

    scores = {}

    for label, vecs in label_to_vecs.items():
        sims = cosine_similarity(q_emb, vecs)[0]
        scores[id2label[label]] = float(np.max(sims))  # or mean

    return dict(sorted(scores.items(), key=lambda x: -x[1]))

# examples

examples = [ 
    "How much to get to new york?",
    "How hot is it?",
    "Where's my phone?",
    "What happened in Chicago?",
    "Where's frank?",
    "FUCK ME",
    "WHat's that pokemon?",
    "Wgat's that pokemon?",
    "liar liar liar liar liar",
    "I wish I could remember that!",
    "Huh?",
    "remind me to call mom tomorrow",
    "I always forget my mom’s birthday",
    "I need METH",
    "say gex",
    "Why is RAM so expensive?",
    "Why is TEX so expensive?",
    "Why is USD so expensive?",
    "What's my credit score?",
    "Make it darker",
    "Play",
    "What was that?" 
]

for e in examples:
  print(e)
  print(classify(e))
  print("="*40)