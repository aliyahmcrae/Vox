# pip install datasets sentence_transformers scikit-learn numpy

import time
import numpy as np

from collections import defaultdict

from datasets import load_dataset

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

from sentence_transformers import SentenceTransformer

print("Loading dataset...")

dataset = load_dataset("DeepPavlov/hwu64")
rows = dataset["train"]

texts = list(rows["utterance"])
labels = np.array(rows["label"])

X_train, X_test, y_train, y_test = train_test_split(
    texts,
    labels,
    test_size=0.20,
    random_state=42,
    stratify=labels
)

print(
    f"train={len(X_train)} "
    f"test={len(X_test)}"
)

# --------------------------------------------------
# EMBEDDING CENTROID MODEL
# --------------------------------------------------

print("\n=== EMBEDDING MODEL ===")

embedder = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

train_emb = embedder.encode(
    X_train,
    normalize_embeddings=True,
    batch_size=256,
    show_progress_bar=True
)

centroids = {}

tmp = defaultdict(list)

for emb, label in zip(
    train_emb,
    y_train
):
    tmp[label].append(emb)

for label, vecs in tmp.items():
    vecs = np.asarray(vecs)

    centroid = vecs.mean(axis=0)
    centroid /= np.linalg.norm(centroid)

    centroids[label] = centroid

start = time.perf_counter()

test_emb = embedder.encode(
    X_test,
    normalize_embeddings=True,
    batch_size=256,
    show_progress_bar=True
)

preds = []

for q in test_emb:

    best_label = None
    best_score = -1

    for label, centroid in centroids.items():

        score = float(
            np.dot(q, centroid)
        )

        if score > best_score:
            best_score = score
            best_label = label

    preds.append(best_label)

elapsed = (
    time.perf_counter()
    - start
)

acc = accuracy_score(
    y_test,
    preds
)

print(
    f"accuracy = {acc:.4f}"
)

print(
    f"test time = {elapsed:.3f}s"
)

print(
    f"per sample = "
    f"{elapsed/len(X_test)*1000:.3f} ms"
)

# --------------------------------------------------
# TF-IDF CENTROID MODEL
# --------------------------------------------------

print("\n=== TFIDF MODEL ===")

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1,2),
    min_df=2
)

X_train_vec = vectorizer.fit_transform(
    X_train
)

centroids = {}

for label in np.unique(y_train):

    idx = np.where(
        y_train == label
    )[0]

    centroid = (
        X_train_vec[idx]
        .mean(axis=0)
    )

    centroid = np.asarray(
        centroid
    ).ravel()

    norm = np.linalg.norm(
        centroid
    )

    if norm:
        centroid /= norm

    centroids[label] = centroid

start = time.perf_counter()

X_test_vec = vectorizer.transform(
    X_test
)

preds = []

for row in X_test_vec:

    q = row.toarray()[0]

    norm = np.linalg.norm(q)

    if norm:
        q /= norm

    best_label = None
    best_score = -1

    for label, centroid in centroids.items():

        score = np.dot(
            q,
            centroid
        )

        if score > best_score:

            best_score = score
            best_label = label

    preds.append(
        best_label
    )

elapsed = (
    time.perf_counter()
    - start
)

acc = accuracy_score(
    y_test,
    preds
)

print(
    f"accuracy = {acc:.4f}"
)

print(
    f"test time = {elapsed:.3f}s"
)

print(
    f"per sample = "
    f"{elapsed/len(X_test)*1000:.3f} ms"
)