BASE_URL = https://s3.magnusfulton.com/shared/labrador
CUES_DIR = labrador/cues

# Piper voice — swap for any voice from https://github.com/rhasspy/piper/blob/master/VOICES.md
PIPER_VOICE = en_US-lessac-medium
PIPER_BASE = https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium

FILES = \
	$(CUES_DIR)/alloy_cue1.wav \
	$(CUES_DIR)/alloy_cue2.wav \
	$(CUES_DIR)/alloy_cue3.wav \
	labrador/secrets.json \
	labrador/$(PIPER_VOICE).onnx \
	labrador/$(PIPER_VOICE).onnx.json

.PHONY: all download clean run

# Download all files
download: $(FILES)

$(CUES_DIR)/alloy_cue1.wav:
	mkdir -p $(CUES_DIR)
	curl -fL -o $@ $(BASE_URL)/cues/alloy_cue1.wav

$(CUES_DIR)/alloy_cue2.wav:
	mkdir -p $(CUES_DIR)
	curl -fL -o $@ $(BASE_URL)/cues/alloy_cue2.wav

$(CUES_DIR)/alloy_cue3.wav:
	mkdir -p $(CUES_DIR)
	curl -fL -o $@ $(BASE_URL)/cues/alloy_cue3.wav

labrador/secrets.json:
	mkdir -p labrador
	curl -fL -o $@ $(BASE_URL)/secrets.json

labrador/$(PIPER_VOICE).onnx:
	mkdir -p labrador
	curl -fL -o $@ $(PIPER_BASE)/$(PIPER_VOICE).onnx

labrador/$(PIPER_VOICE).onnx.json:
	mkdir -p labrador
	curl -fL -o $@ $(PIPER_BASE)/$(PIPER_VOICE).onnx.json

# Remove downloaded files
clean:
	rm -rf labrador

# Run the app (ensures venv exists)
run: download
	. env/bin/activate && python main.py
