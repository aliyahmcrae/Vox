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

download_if "https://s3.magnusfulton.com/shared/labrador/en_US-lessac-medium.onnx" "$dest/en_US-lessac-medium.onnx"
download_if "https://s3.magnusfulton.com/shared/labrador/en_US-lessac-medium.onnx.json" "$dest/en_US-lessac-medium.onnx.json"
download_if "https://s3.magnusfulton.com/shared/labrador/secrets.json" "$dest/secrets.json"

. env/bin/activate

python3 main.py