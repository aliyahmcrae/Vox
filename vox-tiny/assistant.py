# pip install sounddevice python-vlc vosk PyAudio scikit-learn

import json
import queue
import pickle

import sounddevice as sd
import simpleaudio

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

with open("intent_names.txt") as f:
    intent_names = [x.strip() for x in f if x.strip()]
    # preserve order and remove duplicates while keeping the first occurrence
    intent_names = list(dict.fromkeys(intent_names))

# --------------------
# audio playback
# --------------------

current_player = None


def play_intent(intent):
    path = f"intents/{intent}.wav"
    wave = simpleaudio.WaveObject.from_wave_file(path)
    play = wave.play()
    play.wait_done()
    
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

    best_label_index = probs.argmax()

    confidence = float(
        probs[best_label_index]
    )

    best_class = classifier.classes_[best_label_index]

    # If the classifier's class is an integer index into intent_names, use it.
    # Otherwise fall back to the classifier class value as a string.
    intent_name = None
    try:
        label_index = int(best_class)
    except Exception:
        label_index = None

    if label_index is not None and 0 <= label_index < len(intent_names):
        intent_name = intent_names[label_index]
    else:
        # If the class itself is already an intent name, prefer that.
        if str(best_class) in intent_names:
            intent_name = str(best_class)
        else:
            intent_name = str(best_class)

    return (
        intent_name,
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
