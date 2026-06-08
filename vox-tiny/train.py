# pip install datasets sentence_transformers numpy

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from collections import defaultdict
import numpy as np
import pickle

print("Loading dataset...")
dataset = load_dataset("DeepPavlov/hwu64")
train = dataset["train"]
texts = train["utterance"]
labels = train["label"]
model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)
embeddings = model.encode(
    texts,
    normalize_embeddings=True,
    batch_size=256,
    show_progress_bar=True
)
label_to_embeddings = defaultdict(list)
for emb, label in zip(
    embeddings,
    labels
):
    label_to_embeddings[label].append(emb)
intent_centroids = {}
for label, vecs in label_to_embeddings.items():
    vecs = np.asarray(vecs)
    centroid = vecs.mean(axis=0)
    centroid /= np.linalg.norm(centroid)
    intent_centroids[label] = centroid
with open(
    "intent_embeddings.pkl",
    "wb"
) as f:
    pickle.dump(
        intent_centroids,
        f
    )
print("Saved.")
