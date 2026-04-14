# VoxCPM API Server — OpenAI-compatible TTS for aarch64 DGX Spark
# Base image mirrors docker/whisper/Dockerfile in dgx-spark-ai-cluster
# (CUDA 12.8 + cuDNN on Ubuntu 24.04) for proven aarch64 compatibility.
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04

LABEL maintainer="jinwangmok"
LABEL description="VoxCPM2 TTS server exposing OpenAI /v1/audio/speech on aarch64 DGX Spark"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-venv \
        python3.12-dev \
        python3-pip \
        build-essential \
        git \
        curl \
        ffmpeg \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip setuptools wheel

# PyTorch first (CUDA 12.8 wheels). aarch64 wheels are published on the
# default index for recent torch releases; if resolution fails on a given
# host, override TORCH_INDEX_URL at build time.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
RUN pip install --index-url ${TORCH_INDEX_URL} \
        torch==2.5.1 \
        torchaudio==2.5.1 \
    || pip install torch==2.5.1 torchaudio==2.5.1

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Install VoxCPM from source — not on PyPI at the time of writing.
ARG VOXCPM_REF=main
RUN pip install "git+https://github.com/OpenBMB/VoxCPM.git@${VOXCPM_REF}"

WORKDIR /app
COPY server.py /app/server.py

# Unified cache under /data/models/voxcpm matches whisper convention.
ENV VOXCPM_MODEL=openbmb/VoxCPM2 \
    VOXCPM_PORT=9100 \
    HF_HOME=/data/models \
    HF_HUB_CACHE=/data/models/voxcpm \
    VOXCPM_CACHE_DIR=/data/models/voxcpm

EXPOSE 9100

CMD ["python3", "server.py"]
