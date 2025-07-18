# Dockerfile

FROM python:3.11-slim

WORKDIR /app

# 1) system deps
RUN apt-get update \
 && apt-get install -y --no-install-recommends git build-essential \
 && rm -rf /var/lib/apt/lists/*

# 2) copy & install python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3) copy source
COPY src/ src/

WORKDIR /app/src

# default to training pipeline
ENTRYPOINT ["python", "train_pipeline.py"]