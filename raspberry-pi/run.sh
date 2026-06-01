#!/usr/bin/env bash
set -euo pipefail

dest=cache
mkdir -p "$dest"

download_if() {
  local url="$1"
  local dest="$2"
  if [ ! -f "$dest" ]; then
    echo "Downloading $url -> $dest"
    curl -fsSL "$url" -o "$dest"
  fi
}

download_if "https://s3.magnusfulton.com/shared/labrador/secrets.json" "$dest/secrets.json"

moonshinedest="$dest/moonshine/tiny-streaming"
mkdir -p "$moonshinedest"
download_if "https://s3.magnusfulton.com/shared/labrador/moonshine-tiny-streaming-en/adapter.ort" "$moonshinedest/adapter.ort"
download_if "https://s3.magnusfulton.com/shared/labrador/moonshine-tiny-streaming-en/cross_kv.ort" "$moonshinedest/cross_kv.ort"
download_if "https://s3.magnusfulton.com/shared/labrador/moonshine-tiny-streaming-en/decoder_kv.ort" "$moonshinedest/decoder_kv.ort"
download_if "https://s3.magnusfulton.com/shared/labrador/moonshine-tiny-streaming-en/decoder_kv_with_attention.ort" "$moonshinedest/decoder_kv_with_attention.ort"
download_if "https://s3.magnusfulton.com/shared/labrador/moonshine-tiny-streaming-en/encoder.ort" "$moonshinedest/encoder.ort"
download_if "https://s3.magnusfulton.com/shared/labrador/moonshine-tiny-streaming-en/frontend.ort" "$moonshinedest/frontend.ort"
download_if "https://s3.magnusfulton.com/shared/labrador/moonshine-tiny-streaming-en/streaming_config.json" "$moonshinedest/streaming_config.json"
download_if "https://s3.magnusfulton.com/shared/labrador/moonshine-tiny-streaming-en/tokenizer.bin" "$moonshinedest/tokenizer.bin"

if [[ ! -x env/bin/python ]]; then
  python3 -m venv env
  source env/bin/activate
  pip install -r requirements.txt
fi

env/bin/python3 main.py