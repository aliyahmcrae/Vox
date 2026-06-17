#! /bin/bash

rm -rf vosk-model-small-en-us-0.15 intents

wget https://s3.magnusfulton.com/shared/labrador/vosk-model-small-en-us-0.15.zip
unzip vosk-model-small-en-us-0.15.zip
rm vosk-model-small-en-us-0.15.zip

wget https://s3.magnusfulton.com/shared/labrador/intents-spoken-wav.zip
unzip intents-spoken-wav.zip
rm intents-spoken-wav.zip
