FROM python:3.10-slim

# System dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
RUN pip install --no-cache-dir \
    numpy \
    pandas \
    tqdm

WORKDIR /app

# Copy only what we need
COPY extract_openpose_dataset.py /app/extract_openpose_dataset.py

CMD ["python", "extract_openpose_dataset.py"]
