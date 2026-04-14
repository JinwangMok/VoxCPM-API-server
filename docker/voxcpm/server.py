"""
VoxCPM API Server — OpenAI /v1/audio/speech compatible TTS.

Wraps OpenBMB/VoxCPM2 (https://github.com/OpenBMB/VoxCPM) with a FastAPI
surface that matches the subset of the OpenAI audio API needed by common
clients (openai-python, LangChain, LiteLLM, etc.).

Exposed routes:
    GET  /health               liveness/readiness probe
    GET  /v1/models            list the single loaded VoxCPM model
    POST /v1/audio/speech      synthesize speech from text

Environment:
    VOXCPM_MODEL       HF repo id or local path (default openbmb/VoxCPM2)
    VOXCPM_PORT        listen port (default 9100)
    VOXCPM_CACHE_DIR   where HF weights are cached (default /data/models/voxcpm)
    VOXCPM_CFG_VALUE   default classifier-free guidance (default 2.0)
    VOXCPM_TIMESTEPS   default inference timesteps (default 10)
"""

import io
import os
import logging
import threading
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("voxcpm-server")


# --- OpenAI voice name -> VoxCPM voice-design description ------------------
# VoxCPM2 supports "Voice Design": prepending "(description)" to the text
# creates a voice matching that description, no reference audio required.
# We map the six OpenAI preset voices to sensible descriptions so existing
# OpenAI clients work unchanged.
VOICE_PRESETS = {
    "alloy":   "A neutral adult voice, clear and balanced, medium pace",
    "echo":    "A calm adult male voice, warm and measured",
    "fable":   "A British adult male voice, expressive storyteller tone",
    "onyx":    "A deep adult male voice, authoritative and resonant",
    "nova":    "A young adult female voice, bright and friendly",
    "shimmer": "A soft adult female voice, gentle and airy",
}
DEFAULT_VOICE = "alloy"

SUPPORTED_FORMATS = {"wav", "flac", "mp3", "opus", "pcm", "aac"}
# soundfile handles wav/flac/ogg natively; for mp3/opus/aac we fall back to
# ffmpeg via a subprocess pipe. pcm = raw 16-bit little-endian at model SR.
SOUNDFILE_FORMATS = {"wav": "WAV", "flac": "FLAC"}
FFMPEG_FORMATS = {
    "mp3":  ("mp3",  "libmp3lame", "audio/mpeg"),
    "opus": ("ogg",  "libopus",    "audio/ogg"),
    "aac":  ("adts", "aac",        "audio/aac"),
}
CONTENT_TYPES = {
    "wav":  "audio/wav",
    "flac": "audio/flac",
    "mp3":  "audio/mpeg",
    "opus": "audio/ogg",
    "aac":  "audio/aac",
    "pcm":  "application/octet-stream",
}


_model = None
_model_lock = threading.Lock()


def get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from voxcpm import VoxCPM

                model_id = os.environ.get("VOXCPM_MODEL", "openbmb/VoxCPM2")
                cache_dir = os.environ.get("VOXCPM_CACHE_DIR", "/data/models/voxcpm")
                os.makedirs(cache_dir, exist_ok=True)
                logger.info("loading VoxCPM model=%s cache=%s", model_id, cache_dir)
                _model = VoxCPM.from_pretrained(model_id, load_denoiser=False)
                logger.info("VoxCPM ready (sample_rate=%s)", _model.tts_model.sample_rate)
    return _model


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        get_model()
    except Exception:
        logger.exception("failed to load VoxCPM model")
        raise
    yield


app = FastAPI(title="VoxCPM API Server", lifespan=lifespan)


class SpeechRequest(BaseModel):
    model: str = Field(default="voxcpm2")
    input: str = Field(..., min_length=1, max_length=4096)
    voice: str = Field(default=DEFAULT_VOICE)
    response_format: str = Field(default="wav")
    speed: float = Field(default=1.0, ge=0.25, le=4.0)
    # VoxCPM-specific knobs, exposed as optional extensions.
    cfg_value: Optional[float] = None
    inference_timesteps: Optional[int] = None


@app.get("/health")
async def health():
    if _model is None:
        return JSONResponse(status_code=503, content={"status": "loading"})
    return {"status": "ok", "model": os.environ.get("VOXCPM_MODEL", "openbmb/VoxCPM2")}


@app.get("/v1/models")
async def list_models():
    model_id = os.environ.get("VOXCPM_MODEL", "openbmb/VoxCPM2")
    return {
        "object": "list",
        "data": [
            {"id": "voxcpm2", "object": "model", "owned_by": "openbmb", "root": model_id},
        ],
    }


def _encode_audio(wav: np.ndarray, sample_rate: int, fmt: str) -> bytes:
    fmt = fmt.lower()
    if fmt not in SUPPORTED_FORMATS:
        raise HTTPException(status_code=400, detail=f"unsupported response_format: {fmt}")

    # Normalize dtype/shape for encoders.
    if wav.dtype != np.float32:
        wav = wav.astype(np.float32)
    if wav.ndim > 1:
        wav = np.squeeze(wav)

    if fmt == "pcm":
        pcm16 = np.clip(wav, -1.0, 1.0)
        pcm16 = (pcm16 * 32767.0).astype("<i2")
        return pcm16.tobytes()

    if fmt in SOUNDFILE_FORMATS:
        buf = io.BytesIO()
        sf.write(buf, wav, sample_rate, format=SOUNDFILE_FORMATS[fmt])
        return buf.getvalue()

    # ffmpeg-mediated formats.
    import subprocess
    container, codec, _ = FFMPEG_FORMATS[fmt]
    pcm16 = np.clip(wav, -1.0, 1.0)
    pcm16 = (pcm16 * 32767.0).astype("<i2").tobytes()
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-f", "s16le", "-ar", str(sample_rate), "-ac", "1", "-i", "pipe:0",
        "-c:a", codec, "-f", container, "pipe:1",
    ]
    try:
        proc = subprocess.run(cmd, input=pcm16, capture_output=True, check=True)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail="ffmpeg not available in image") from exc
    except subprocess.CalledProcessError as exc:
        logger.error("ffmpeg failed: %s", exc.stderr.decode(errors="replace"))
        raise HTTPException(status_code=500, detail="audio encoding failed")
    return proc.stdout


@app.post("/v1/audio/speech")
async def speech(req: SpeechRequest):
    model = get_model()

    voice_key = (req.voice or DEFAULT_VOICE).lower()
    description = VOICE_PRESETS.get(voice_key)
    if description is None:
        raise HTTPException(
            status_code=400,
            detail=f"unknown voice '{req.voice}'. supported: {sorted(VOICE_PRESETS)}",
        )

    # Speed control is expressed via VoxCPM's natural-language steering.
    pace_clause = ""
    if req.speed <= 0.8:
        pace_clause = ", slower pace"
    elif req.speed >= 1.25:
        pace_clause = ", faster pace"

    prompt_text = f"({description}{pace_clause}){req.input}"

    cfg_value = req.cfg_value if req.cfg_value is not None else float(
        os.environ.get("VOXCPM_CFG_VALUE", "2.0")
    )
    timesteps = req.inference_timesteps if req.inference_timesteps is not None else int(
        os.environ.get("VOXCPM_TIMESTEPS", "10")
    )

    try:
        wav = model.generate(
            text=prompt_text,
            cfg_value=cfg_value,
            inference_timesteps=timesteps,
        )
    except Exception:
        logger.exception("voxcpm generate failed")
        raise HTTPException(status_code=500, detail="speech synthesis failed")

    sample_rate = model.tts_model.sample_rate
    audio_bytes = _encode_audio(np.asarray(wav), sample_rate, req.response_format)
    return Response(
        content=audio_bytes,
        media_type=CONTENT_TYPES.get(req.response_format.lower(), "application/octet-stream"),
    )


if __name__ == "__main__":
    port = int(os.environ.get("VOXCPM_PORT", "9100"))
    logger.info("starting voxcpm server on port %s", port)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
