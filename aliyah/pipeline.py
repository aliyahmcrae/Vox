import time
from stt import listen_and_transcribe
from intent import classify
from responder import respond
from tts import speak

RECORD_SECONDS = 5
SHOW_DEBUG = True


def log(msg: str):
    print(f"[Vox] {msg}")


def run_once(history: list[dict]) -> list[dict]:
    log("Listening...")
    text = listen_and_transcribe(duration=RECORD_SECONDS)
    log(f"Heard: \"{text}\"")

    if not text.strip():
        log("Nothing detected, skipping.")
        return history

    intent_result = classify(text)
    if SHOW_DEBUG:
        log(f"Intent: {intent_result}")

    if not intent_result["should_respond"]:
        log(f"Intent '{intent_result['intent']}' — staying silent.")
        return history

    log("Generating response...")
    response = respond(text, history=history)
    log(f"Response: \"{response}\"")

    speak(response)

    history.append({"role": "user", "content": text})
    history.append({"role": "assistant", "content": response})

    if len(history) > 10:
        history = history[-10:]

    return history


def main():
    log("Starting Vox. Press Ctrl+C to quit.")
    log("=" * 50)
    history = []
    while True:
        try:
            history = run_once(history)
            time.sleep(0.3)
        except KeyboardInterrupt:
            log("Shutting down.")
            break
        except Exception as e:
            log(f"Error: {e}")
            continue


if __name__ == "__main__":
    main()
