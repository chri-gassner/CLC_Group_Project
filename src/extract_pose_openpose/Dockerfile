FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    numpy \
    pandas \
    tqdm \
    psutil

WORKDIR /app

COPY extract_pose_openpose.py /app/extract_pose_openpose.py

# Optional: ensure output dirs exist (also created by script)
RUN mkdir -p /app/output /app/output/metrics/openpose

CMD ["python", "extract_pose_openpose.py"]
