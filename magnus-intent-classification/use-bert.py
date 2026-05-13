import torch, requests, os
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MODEL_DIR = "4_27_intent_bert"
os.makedirs(MODEL_DIR, exist_ok = True)

def get_file(name):
  if not os.path.isfile(f"{MODEL_DIR}/{name}"):
    with requests.get(f"https://s3.magnusfulton.com/shared/labrador/{MODEL_DIR}/{name}", stream=True) as r:
      with open(f"{MODEL_DIR}/{name}", "wb") as f:
        for chunk in r.iter_content(chunk_size=8192): 
          f.write(chunk)

for file in ("config.json", "model.safetensors", "tokenizer_config.json", "tokenizer.json", "training_args.bin"):
  get_file(file)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

with open("intent_names.txt") as f:
  intent_names = sorted(i.strip() for i in set(f))

id2label = {i: name for i, name in enumerate(intent_names)}
label2id = {v: k for k, v in id2label.items()}

def classify(text):
  inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
  outputs = model(**inputs)

  probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]

  scores = {
    id2label[i]: float(probs[i])
    for i in range(len(id2label))
  }

  return dict(sorted(scores.items(), key=lambda x: -x[1]))

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