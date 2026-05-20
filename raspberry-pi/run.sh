#!/usr/bin/env bash
set -euo pipefail

dest=cache
mkdir -p "$dest"
mkdir -p "$dest/tiny"

download_if() {
  local url="$1"
  local dest="$2"
  if [ ! -f "$dest" ]; then
    echo "Downloading $url -> $dest"
    curl -fsSL "$url" -o "$dest"
  fi
}

download_if "https://s3.magnusfulton.com/shared/labrador/faster-whisper-tiny/model.bin" "$dest/tiny/model.bin"
download_if "https://s3.magnusfulton.com/shared/labrador/faster-whisper-tiny/vocabulary.txt" "$dest/tiny/vocabulary.txt"
download_if "https://s3.magnusfulton.com/shared/labrador/faster-whisper-tiny/tokenizer.json" "$dest/tiny/tokenizer.json"
download_if "https://s3.magnusfulton.com/shared/labrador/secrets.json" "$dest/secrets.json"

. env/bin/activate

python3 main.py