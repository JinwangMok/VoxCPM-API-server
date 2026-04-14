# VoxCPM API Server — OpenAI-compatible TTS for aarch64 DGX Spark (GB10).
#
# Uses the lean nvidia/cuda base image and installs torch/torchvision/
# torchaudio from download.pytorch.org/whl/cu130 — the approach the
# PyTorch maintainers recommend for DGX Spark (GB10, sm_121). NGC's
# bundled pytorch containers were tried first but:
#   - 25.02 ships no CUDA kernels for Blackwell (VoxCPM fell to CPU)
#   - 25.12 has a known torchaudio incompatibility on DGX Spark
#   - 26.02 ships no torchaudio at all (unresolved upstream issue)
#
# References:
#   https://discuss.pytorch.org/t/dgx-spark-gb10-cuda-13-0-python-3-12-sm-121/223744
#   https://forums.developer.nvidia.com/t/incompatibility-of-torchaudio-in-ngc-pytorch-container-25-12-on-dgx-spark-blackwell-gb10/357745
FROM nvidia/cuda:13.0.0-cudnn-runtime-ubuntu24.04

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

# Install torch stack from PyTorch's cu130 aarch64 index. These wheels
# are the only officially-published ARM + CUDA 13 torch binaries, and
# per the PyTorch forum they support DGX Spark (sm_121 is binary-
# compatible with sm_120, which the wheels include). Pin major versions
# loosely and rely on the index to pick a compatible tuple.
RUN pip install --index-url https://download.pytorch.org/whl/cu130 \
        torch \
        torchvision \
        torchaudio

# Non-torch runtime deps for the FastAPI surface.
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# VoxCPM runtime deps, *excluding* torch/torchaudio/gradio so pip cannot
# reinstall them from PyPI (which would clobber our cu130 torch with
# CPU-only wheels). gradio is dropped because the API server has no
# need for the demo UI. torchcodec and funasr pull in additional audio
# stacks that voxcpm actually uses at runtime.
RUN pip install \
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

# Install VoxCPM itself with --no-deps so pip does not re-resolve torch.
ARG VOXCPM_REF=main
RUN pip install --no-deps \
        "git+https://github.com/OpenBMB/VoxCPM.git@${VOXCPM_REF}"

# Sanity check at build time: installed torch must be a CUDA build and
# torchaudio must import cleanly. Fails the build on regression instead
# of silently degrading at runtime.
RUN python3 -c "import torch, torchaudio, sys; v = torch.version.cuda; print('torch', torch.__version__, 'cuda', v, 'torchaudio', torchaudio.__version__); sys.exit(0 if v else 1)"

WORKDIR /app
COPY server.py /app/server.py

ENV VOXCPM_MODEL=openbmb/VoxCPM2 \
    VOXCPM_PORT=9100 \
    HF_HOME=/data/models \
    HF_HUB_CACHE=/data/models/voxcpm \
    VOXCPM_CACHE_DIR=/data/models/voxcpm

EXPOSE 9100

CMD ["python3", "server.py"]
