# docker run -it ubuntu:20.04 /bin/bash
FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# apt-get update && apt-get install -y build-essential cmake git wget unzip sudo
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    unzip \
    sudo

# git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose.git
RUN git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose.git

# cd openpose
WORKDIR /openpose

# chmod +x scripts/ubuntu/install_deps.sh
RUN chmod +x scripts/ubuntu/install_deps.sh

# ./scripts/ubuntu/install_deps.sh
RUN ./scripts/ubuntu/install_deps.sh

# mkdir build && cd build
RUN mkdir build

# cd build
WORKDIR /openpose/build

# install updates and opencv
RUN apt-get update && apt-get install -y libopencv-dev

# cmake openpose
RUN cmake -DGPU_MODE=CPU_ONLY -DBUILD_PYTHON=OFF \
  -DDOWNLOAD_BODY_25_MODEL=ON \
  -DDOWNLOAD_FACE_MODEL=OFF \
  -DDOWNLOAD_HAND_MODEL=OFF \
  -DDOWNLOAD_BODY_COCO_MODEL=OFF \
  -DDOWNLOAD_BODY_MPI_MODEL=OFF \
  ..

# make openpose
RUN make -j"$(nproc)"

# Copy and Install requirements
WORKDIR /app
COPY src/extract_pose_openpose/docker/requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r /app/requirements.txt


# COPY src/data/ /app/src/data/
COPY src/extract_pose_openpose/extract_pose_openpose.py /app/src/extract_pose_openpose/extract_pose_openpose.py
COPY src/utils/make_manifest.py /app/src/extract_pose_openpose/make_manifest.py
COPY src/extract_pose_openpose/docker/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# set working directory
WORKDIR /app/src/extract_pose_openpose

# entrypoint script
ENTRYPOINT ["/app/entrypoint.sh"]
