# VoxCPM API Server — OpenAI-compatible TTS for aarch64 DGX Spark
#
# Uses the NVIDIA NGC PyTorch container as the base. This is the only
# supported path for CUDA-enabled torch on aarch64 — download.pytorch.org
# publishes aarch64 wheels but only in CPU flavor, so a plain
# `nvidia/cuda` base image ends up running VoxCPM on the CPU.
#
# NGC tags follow YY.MM. 25.02 ships Python 3.12 + torch 2.6 + CUDA 12.8,
# which matches the DGX Spark host software stack. Override with
# --build-arg BASE_IMAGE=... if you need a different release.
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.02-py3
FROM ${BASE_IMAGE}

LABEL maintainer="jinwangmok"
LABEL description="VoxCPM2 TTS server exposing OpenAI /v1/audio/speech on aarch64 DGX Spark"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Runtime system deps. NGC already provides python3, torch, torchaudio,
# CUDA, cuDNN, and build toolchain, so we only add audio IO + git.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        ffmpeg \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install server deps into NGC's system Python (no venv — NGC's torch
# lives in /usr/local/lib/python*/dist-packages and isolating breaks it).
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Install VoxCPM from source — not on PyPI at the time of writing.
ARG VOXCPM_REF=main
RUN pip install --no-cache-dir "git+https://github.com/OpenBMB/VoxCPM.git@${VOXCPM_REF}"

WORKDIR /app
COPY server.py /app/server.py

# Unified cache under /data/models/voxcpm matches the whisper convention
# in dgx-spark-ai-cluster.
ENV VOXCPM_MODEL=openbmb/VoxCPM2 \
    VOXCPM_PORT=9100 \
    HF_HOME=/data/models \
    HF_HUB_CACHE=/data/models/voxcpm \
    VOXCPM_CACHE_DIR=/data/models/voxcpm

EXPOSE 9100

CMD ["python3", "server.py"]
