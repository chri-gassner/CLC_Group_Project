FROM python:3.10-slim

# System dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*


# Python dependencies
RUN pip install --no-cache-dir \
    protobuf==3.20.3 \
    mediapipe==0.10.11 \
    opencv-python==4.7.0.72 \
    numpy \
    pandas \
    tqdm


WORKDIR /app

COPY mediapipe_dataset_creation.py /app/mediapipe_dataset_creation.py

CMD ["python", "mediapipe_dataset_creation.py"]
