FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    git \
    build-essential \
    curl \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

COPY main.py .

ENV WHISPER_MODEL=tiny.en
ENV ORACLE_WS=wss://api.magnusfulton.com/cse481/ws/pi
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('tiny.en', device='cpu', compute_type='int8')"

RUN mkdir -p /root/.cache

CMD ["python", "main.py"]