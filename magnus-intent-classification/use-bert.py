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