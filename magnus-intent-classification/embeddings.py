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
    "Remind me to call Sarah at 6pm",
    "What’s the weather like tomorrow?",
    "Set an alarm for 7 in the morning",
    "Add lunch with John to my calendar on Friday",
    "Send a text to Mom saying I’ll be late",
    "Flip a coin for me",
    "Take a note: buy milk and eggs",
    "When is Christmas this year?",
    "What happened in World War 2?",
    "Show me a recipe for chocolate cake",
    "uh can you like remind me about that thing later",
    "weather tmrw??",
    "wake me up at 8",
    "schedule dentist next week idk when",
    "text alex “on my way”",
    "heads or tails",
    "write this down: don’t forget the keys",
    "when’s my birthday again",
    "who was the first president of the US",
    "how do I make pancakes",
    "I always forget to water the plants",
    "is it gonna rain or what",
    "alarm. 6am. don’t let me die.",
    "put meeting on calendar tomorrow afternoon",
    "call dad",
    "pick a random number or something",
    "note this: meeting moved to Thursday",
    "is thanksgiving next week",
    "why did the roman empire fall",
    "easy pasta recipe pls",
    "remind me maybe later idk",
    "hot outside?",
    "set alarm for like… early",
    "add something to my schedule",
    "message her",
    "do a coin flip thing",
    "save this thought somewhere",
    "what day is halloween",
    "tell me about black holes",
    "I need food ideas",
]

for e in examples:
  print(e)
  print(classify(e))
  print("="*40)