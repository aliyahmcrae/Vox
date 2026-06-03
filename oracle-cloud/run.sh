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

if [[ ! -x env/bin/python ]]; then
  python3 -m venv env
  source env/bin/activate
  pip install -r requirements.txt
fi

env/bin/python3 main.py