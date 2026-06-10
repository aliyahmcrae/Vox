# pip install sounddevice python-vlc vosk PyAudio scikit-learn

import json
import queue
import pickle

import sounddevice as sd
import vlc

from vosk import Model
from vosk import KaldiRecognizer

SAMPLE_RATE = 48000
BLOCK_SIZE = 4000

audio_q = queue.Queue()

# --------------------
# load model
# --------------------

with open(
    "intent_model.pkl",
    "rb"
) as f:
    vectorizer, classifier = pickle.load(f)

# --------------------
# load intent names
# --------------------

with open(
    "intent_names.txt"
) as f:
    intent_names = list(
        set(
            x.strip()
            for x in f
            if x.strip()
        )
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

        current_player = vlc.MediaPlayer(
            path
        )

        current_player.play()

    except Exception as e:

        print(e)
        print(
            "Would've played:",
            path
        )

# --------------------
# intent classifier
# --------------------


def classify(text):

    x = vectorizer.transform(
        [text]
    )

    probs = classifier.predict_proba(
        x
    )[0]

    best_label = probs.argmax()

    confidence = float(
        probs[best_label]
    )

    return (
        intent_names[classifier.classes_[best_label]],
        confidence
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
        "vosk-model-small-en-us-0.15"
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

        print(
            "Listening..."
        )

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
