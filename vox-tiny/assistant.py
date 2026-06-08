# pip install numpy sounddevice python-vlc vosk

import json
import queue
import pickle
import numpy as np
import sounddevice as sd
import vlc

from vosk import Model
from vosk import KaldiRecognizer

from sentence_transformers import (
    SentenceTransformer
)

SAMPLE_RATE = 48000
BLOCK_SIZE = 4000

audio_q = queue.Queue()

# --------------------
# load intent names
# --------------------
with open(
    "intent_names.txt"
) as f:
    intent_names = [
        x.strip()
        for x in f
        if x.strip()
    ]

# --------------------
# load centroids
# --------------------
with open(
    "intent_embeddings.pkl",
    "rb"
) as f:
    intent_centroids = pickle.load(f)

# --------------------
# embedding model
# --------------------
embedder = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# --------------------
# audio playback
# --------------------
current_player = None


def play_intent(intent):
    global current_player
    path = f"intents/{intent}.mp3"
    try:
        if current_player:
            current_player.stop()
        current_player = vlc.MediaPlayer(path)
        current_player.play()
    except Exception as e:
        print(e)
        print("Would've played:", path)

# --------------------
# intent classifier
# --------------------


def classify(text):
    q = embedder.encode(
        [text],
        normalize_embeddings=True
    )[0]
    best_score = -1
    best_label = None
    for label, centroid in \
            intent_centroids.items():
        score = float(
            np.dot(q, centroid)
        )
        if score > best_score:
            best_score = score
            best_label = label
    return (
        intent_names[best_label],
        best_score
    )

# --------------------
# mic callback
# --------------------


def audio_callback(
    indata,
    frames,
    time_info,
    status
):
    if status:
        print(status)
    audio_q.put(
        bytes(indata)
    )

# --------------------
# main
# --------------------


def main():
    model = Model(
        "model"
    )
    recognizer = KaldiRecognizer(
        model,
        SAMPLE_RATE
    )
    with sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_SIZE,
        dtype="int16",
        channels=1,
        callback=audio_callback
    ):
        print("Listening...")
        while True:
            data = audio_q.get()
            if recognizer.AcceptWaveform(
                data
            ):
                result = json.loads(
                    recognizer.Result()
                )
                text = result.get(
                    "text",
                    ""
                )
                if not text:
                    continue
                intent, score = classify(
                    text
                )
                print(
                    f"{text!r}"
                )
                print(
                    f"→ {intent} "
                    f"({score:.3f})"
                )
                if score > 0.45:
                    play_intent(
                        intent
                    )


if __name__ == "__main__":
    main()
