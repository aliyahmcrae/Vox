BASE_URL = https://s3.magnusfulton.com/shared/labrador
CUES_DIR = labrador/cues

FILES = \
	$(CUES_DIR)/alloy_cue1.wav \
	$(CUES_DIR)/alloy_cue2.wav \
	$(CUES_DIR)/alloy_cue3.wav \
	labrador/secrets.json

.PHONY: all download clean run

# Download all files
download: $(FILES)

$(CUES_DIR)/alloy_cue1.wav:
	mkdir -p $(CUES_DIR)
	wget -O $@ $(BASE_URL)/cues/alloy_cue1.wav

$(CUES_DIR)/alloy_cue2.wav:
	mkdir -p $(CUES_DIR)
	wget -O $@ $(BASE_URL)/cues/alloy_cue2.wav

$(CUES_DIR)/alloy_cue3.wav:
	mkdir -p $(CUES_DIR)
	wget -O $@ $(BASE_URL)/cues/alloy_cue3.wav

labrador/secrets.json:
	mkdir -p labrador
	wget -O $@ $(BASE_URL)/secrets.json

# Remove downloaded files
clean:
	rm -rf labrador

# Run the app (ensures venv exists)
run: download
	. env/bin/activate && python main.py
