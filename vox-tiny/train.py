from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

dataset = load_dataset(
    "DeepPavlov/hwu64"
)

texts = dataset["train"]["utterance"]
labels = dataset["train"]["label"]

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2
)

X = vectorizer.fit_transform(
    texts
)

classifier = LogisticRegression(
    max_iter=1000,
    n_jobs=-1
)

classifier.fit(
    X,
    labels
)

with open(
    "intent_model.pkl",
    "wb"
) as f:

    pickle.dump(
        (vectorizer, classifier),
        f
    )