# syntax=docker/dockerfile:1
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1 \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# requirements liegen im docker-Ordner
COPY src/extract_pose_mediapipe/docker/requirements.txt /app/requirements.txt
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r /app/requirements.txt

# nur Pipeline-Code (kein data/, kein output/)
COPY src/extract_pose_mediapipe /app/src/extract_pose_mediapipe
COPY src/utils/make_manifest.py /app/src/extract_pose_mediapipe/make_manifest.py

ENV PYTHONUNBUFFERED=1

COPY src/extract_pose_mediapipe/docker/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# wichtig für dein MANIFEST="manifest.csv"
WORKDIR /app/src/extract_pose_mediapipe

ENTRYPOINT ["/app/entrypoint.sh"]
