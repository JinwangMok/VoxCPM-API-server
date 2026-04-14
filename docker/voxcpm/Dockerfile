# VoxCPM API Server — OpenAI-compatible TTS for aarch64 DGX Spark
#
# Uses the NVIDIA NGC PyTorch container as the base. This is the only
# supported path for CUDA-enabled torch on aarch64 — download.pytorch.org
# publishes aarch64 wheels but only in CPU flavor, so a plain
# `nvidia/cuda` base image ends up running VoxCPM on the CPU.
#
# NGC tags follow YY.MM. 26.02-py3 is the first confirmed-working tag on
# DGX Spark (GB10 / Blackwell, compute capability sm_121). NGC 25.02 does
# not ship CUDA kernels for Blackwell and degrades VoxCPM to the CPU.
# Per NVIDIA PyTorch maintainers: "all of our ARM binaries built with
# CUDA >= 12.8 support DGX Spark already" because sm_121 is binary
# compatible with sm_120, which NGC 25.12+ includes. 25.12 has a known
# torchaudio incompatibility on DGX Spark, so 26.02 is the safer default.
#
# Override with --build-arg BASE_IMAGE=... if you need a different release.
#
# References:
#   https://discuss.pytorch.org/t/dgx-spark-gb10-cuda-13-0-python-3-12-sm-121/223744
#   https://forums.developer.nvidia.com/t/incompatibility-of-torchaudio-in-ngc-pytorch-container-25-12-on-dgx-spark-blackwell-gb10/357745
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.02-py3
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

# Install server deps first.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Install VoxCPM's runtime deps *without* torch/torchaudio — NGC ships
# a CUDA-enabled torch in /opt/venv and a transitive reinstall from
# PyPI would clobber it with a CPU-only aarch64 wheel, which silently
# sends VoxCPM to the CPU. `gradio` is also dropped: the API server has
# no need for the Gradio demo UI and it drags in a heavy dep tree.
RUN pip install --no-cache-dir \
        "transformers>=4.36.2" \
        einops \
        inflect \
        addict \
        wetext \
        "modelscope>=1.22.0" \
        "datasets>=3,<4" \
        huggingface-hub \
        pydantic \
        tqdm \
        simplejson \
        sortedcontainers \
        librosa \
        matplotlib \
        funasr \
        argbind \
        safetensors \
        torchcodec

# Install VoxCPM from source with --no-deps so pip does not re-resolve
# torch/torchaudio/gradio. Not on PyPI at the time of writing.
ARG VOXCPM_REF=main
RUN pip install --no-cache-dir --no-deps \
        "git+https://github.com/OpenBMB/VoxCPM.git@${VOXCPM_REF}"

# Sanity check: fail the build if the installed torch was not compiled
# with CUDA. torch.cuda.is_available() would always be False at build
# time (no GPU visible to buildkit), so we check torch.version.cuda —
# which is None for CPU wheels and a string like "12.8" for NGC builds.
RUN python3 -c "import torch, sys; v = torch.version.cuda; print('torch', torch.__version__, 'cuda', v); sys.exit(0 if v else 1)"

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
