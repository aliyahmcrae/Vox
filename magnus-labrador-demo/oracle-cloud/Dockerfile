FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    wget \
    libsndfile1 \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

RUN wget https://s3.magnusfulton.com/shared/labrador/en_US-lessac-medium.onnx
RUN wget https://s3.magnusfulton.com/shared/labrador/secrets.json

COPY main.py .

EXPOSE 8765

CMD ["python", "main.py"]