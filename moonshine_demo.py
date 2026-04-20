from moonshine_voice import (
    MicTranscriber,
    TranscriptEventListener,
    get_model_for_language,
)

# Download/load model (cached after first run)
model_path, model_arch = get_model_for_language("en")

# Create transcriber (handles mic + VAD + streaming)
mic = MicTranscriber(
    model_path=model_path,
    model_arch=model_arch
)

# Event listener for real-time updates
class Listener(TranscriptEventListener):

    def on_line_started(self, event):
        print("\n[START]", event.line.text)

    def on_line_text_changed(self, event):
        print("[LIVE]", event.line.text, end="\r")

    def on_line_completed(self, event):
        print("\n[FINAL]", event.line.text)

# Attach listener
listener = Listener()
mic.add_listener(listener)

# Start transcription
mic.start()

input("Listening... Push enter to stop!")
