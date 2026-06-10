import numpy as np
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from datasets import load_dataset

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

print("\n=== TFIDF + LOGISTIC REGRESSION ===")

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2
)

X_train_vec = vectorizer.fit_transform(
    X_train
)

X_test_vec = vectorizer.transform(
    X_test
)

clf = LogisticRegression(
    max_iter=1000,
    n_jobs=-1
)

print("Training...")
clf.fit(
    X_train_vec,
    y_train
)

start = time.perf_counter()

preds = clf.predict(
    X_test_vec
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
    f"{elapsed / len(X_test) * 1000:.3f} ms"
)