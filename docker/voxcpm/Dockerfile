# VoxCPM API Server — OpenAI-compatible TTS for aarch64 DGX Spark (GB10).
#
# Base: nvidia/cuda 13.0.2 cudnn-runtime — confirmed linux/arm64 tag on
# Docker Hub. Provides libcudart.so.13, libcudnn9, libcublas etc. needed
# by torch cu130 runtime.
#
# Torch stack from download.pytorch.org/whl/cu130. Verified aarch64
# py3.12 wheels exist:
#   torch-2.11.0+cu130-cp312-cp312-manylinux_2_28_aarch64.whl
#   torchaudio-2.11.0+cu130-cp312-cp312-manylinux_2_28_aarch64.whl
#
# Why not NGC pytorch base:
#   - 25.02 → no Blackwell kernels, silent CPU fallback
#   - 25.12 → known torchaudio incompatibility on GB10
#   - 26.02 → ships no torchaudio at all
#
# torchcodec is deliberately NOT installed: VoxCPM only imports it
# lazily in the reference-audio voice-cloning path, which the OpenAI
# Voice Design preset mapping never triggers. Avoiding it sidesteps
# torchcodec 0.8+'s hard dependency on libnvcuvid.so (not in cudnn-runtime).
#
# References:
#   https://discuss.pytorch.org/t/dgx-spark-gb10-cuda-13-0-python-3-12-sm-121/223744
#   https://forums.developer.nvidia.com/t/incompatibility-of-torchaudio-in-ngc-pytorch-container-25-12-on-dgx-spark-blackwell-gb10/357745
#   https://forums.developer.nvidia.com/t/cant-install-torch-torchaudio-torchcodec/348660
FROM nvidia/cuda:13.0.2-cudnn-runtime-ubuntu24.04

LABEL maintainer="jinwangmok"
LABEL description="VoxCPM2 TTS server exposing OpenAI /v1/audio/speech on aarch64 DGX Spark"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_PREFER_BINARY=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# --- System layer ---------------------------------------------------------
# python 3.12 + dev headers: cp312 wheel match and any source-built dep.
# build-essential + pkg-config: pynini / pyahocorasick / psutil source builds.
# ffmpeg + libsndfile1 + libsox3 + audio codec runtimes: torchaudio/librosa IO.
# libgomp1 + libopenblas0: numba / librosa runtime link targets.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-venv \
        python3.12-dev \
        python3-pip \
        build-essential \
        pkg-config \
        git \
        curl \
        ca-certificates \
        ffmpeg \
        libsndfile1 \
        libsox3 \
        libflac12t64 \
        libvorbis0a \
        libvorbisenc2 \
        libvorbisfile3 \
        libogg0 \
        libopus0 \
        libmp3lame0 \
        libgomp1 \
        libopenblas0 \
    && rm -rf /var/lib/apt/lists/*

# Guard: cp312 wheels require Python 3.12.x. Fail build loudly on drift.
RUN python3.12 --version | grep -qE "^Python 3\.12\." \
    || (echo "FATAL: expected Python 3.12.x" && exit 1)

RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip==24.3.1 setuptools wheel

# --- Torch stack ---------------------------------------------------------
# Pin literal +cu130 local version to prevent any accidental PyPI pickup.
RUN pip install --index-url https://download.pytorch.org/whl/cu130 \
        torch==2.11.0+cu130 \
        torchaudio==2.11.0+cu130

# Lock torch/torchaudio for every subsequent pip install. PIP_CONSTRAINT
# env var applies globally, so any transitive resolution that would
# re-evaluate torch is forced back to the exact cu130 wheels.
# PIP_EXTRA_INDEX_URL lets those constraints resolve against the cu130
# index while everything else continues to come from PyPI.
RUN printf 'torch==2.11.0+cu130\ntorchaudio==2.11.0+cu130\n' > /etc/pip-constraints.txt
ENV PIP_CONSTRAINT=/etc/pip-constraints.txt \
    PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu130

# --- FastAPI server layer ------------------------------------------------
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# --- VoxCPM non-torch runtime deps ---------------------------------------
# Exclusions vs upstream pyproject.toml:
#   torch, torchaudio  → already installed above, locked via PIP_CONSTRAINT
#   torchcodec         → lazy import only, not needed for Voice Design
#   gradio             → demo UI, server does not need it
#   spaces             → HF Spaces helper, not used off-platform
#   pydantic           → already pinned in requirements.txt
RUN pip install \
        "transformers>=4.36.2" \
        einops \
        inflect \
        addict \
        wetext \
        "modelscope>=1.22.0" \
        "datasets>=3,<4" \
        "huggingface-hub>=0.26,<1" \
        tqdm \
        simplejson \
        sortedcontainers \
        librosa \
        matplotlib \
        funasr \
        argbind \
        safetensors

# --- VoxCPM itself -------------------------------------------------------
# --no-deps so VoxCPM's own pyproject cannot re-resolve torch/torchcodec/etc.
ARG VOXCPM_REF=main
RUN pip install --no-deps \
        "git+https://github.com/OpenBMB/VoxCPM.git@${VOXCPM_REF}"

# --- Build-time verification --------------------------------------------
# We deliberately skip `pip check` here: because VoxCPM was installed
# with --no-deps, pip check will (correctly) complain that gradio, spaces,
# and torchcodec are listed in VoxCPM's metadata but not installed. Those
# exclusions are intentional — see comments on the Layer C install above.
# The gates we actually care about are the import assertions below.
#
# The install step may surface a warning like
#   `nvidia-cusparselt-cu13 0.8.0 is not supported on this platform`
# That is a pip metadata quirk from the cu130 wheel bundle — torch uses
# cusparselt only for specific sparse ops and the wheel still imports
# fine on aarch64. If `import torch` succeeds below, we are OK.
RUN python3 -c "\
import torch, torchaudio; \
assert torch.__version__ == '2.11.0+cu130', 'FATAL torch drift: ' + torch.__version__; \
assert torchaudio.__version__ == '2.11.0+cu130', 'FATAL torchaudio drift: ' + torchaudio.__version__; \
assert torch.version.cuda is not None, 'FATAL torch has no CUDA bindings'; \
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'torchaudio', torchaudio.__version__); \
print('build sanity check ok')"
RUN python3 -c "from voxcpm import VoxCPM; print('voxcpm import ok')"

WORKDIR /app
COPY server.py /app/server.py

# Align HF_HOME with HF_HUB_CACHE so snapshot_download sees one cache root.
ENV VOXCPM_MODEL=openbmb/VoxCPM2 \
    VOXCPM_PORT=9100 \
    HF_HOME=/data/models/voxcpm \
    HF_HUB_CACHE=/data/models/voxcpm \
    VOXCPM_CACHE_DIR=/data/models/voxcpm

EXPOSE 9100

# Runtime CUDA gate: if torch cannot see a GPU at container start, exit
# non-zero instead of silently serving on CPU. Uses absolute python path
# to avoid PATH surprises on exec.
CMD ["/opt/venv/bin/python3", "-c", "\
import sys, torch; \
available = torch.cuda.is_available(); \
print('runtime cuda.is_available =', available, flush=True); \
sys.exit(87) if not available else None; \
print('runtime device =', torch.cuda.get_device_name(0), flush=True); \
import os; os.execv('/opt/venv/bin/python3', ['/opt/venv/bin/python3', '/app/server.py'])"]
