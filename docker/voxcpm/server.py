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
import logging
import os
import struct
import tempfile
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field, ValidationError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("voxcpm-server")


# --- OpenAI voice name -> VoxCPM voice-design description ------------------
# VoxCPM2 supports "Voice Design": prepending "(description)" to the text
# creates a voice matching that description, no reference audio required.
# We map the six OpenAI preset voices to sensible descriptions so existing
# OpenAI clients work unchanged.
VOICE_PRESETS = {
    "alloy": "A neutral adult voice, clear and balanced, medium pace",
    "echo": "A calm adult male voice, warm and measured",
    "fable": "A British adult male voice, expressive storyteller tone",
    "onyx": "A deep adult male voice, authoritative and resonant",
    "nova": "A young adult female voice, bright and friendly",
    "shimmer": "A soft adult female voice, gentle and airy",
}
DEFAULT_VOICE = "alloy"
REFERENCE_GAP_MS = 120

SUPPORTED_FORMATS = {"wav", "flac", "mp3", "opus", "pcm", "aac"}
# soundfile handles wav/flac/ogg natively; for mp3/opus/aac we fall back to
# ffmpeg via a subprocess pipe. pcm = raw 16-bit little-endian at model SR.
SOUNDFILE_FORMATS = {"wav": "WAV", "flac": "FLAC"}
FFMPEG_FORMATS = {
    "mp3": ("mp3", "libmp3lame", "audio/mpeg"),
    "opus": ("ogg", "libopus", "audio/ogg"),
    "aac": ("adts", "aac", "audio/aac"),
}
CONTENT_TYPES = {
    "wav": "audio/wav",
    "flac": "audio/flac",
    "mp3": "audio/mpeg",
    "opus": "audio/ogg",
    "aac": "audio/aac",
    "pcm": "application/octet-stream",
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
    voice: Optional[str] = None
    response_format: str = Field(default="wav")
    speed: float = Field(default=1.0, ge=0.25, le=4.0)
    # VoxCPM-specific knobs, exposed as optional extensions.
    cfg_value: Optional[float] = None
    inference_timesteps: Optional[int] = None
    style_prompt: Optional[str] = None
    prompt_text: Optional[str] = None
    # Streaming extension. When true, the response uses HTTP chunked
    # transfer encoding and yields audio as VoxCPM produces it, so the
    # client starts receiving bytes before synthesis is complete.
    # Only pcm and wav response formats are supported for streaming —
    # lossy formats need a full buffer pass to encode. Default false to
    # preserve OpenAI-compatible buffered behavior.
    stream: bool = False


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
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "s16le",
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        "-i",
        "pipe:0",
        "-c:a",
        codec,
        "-f",
        container,
        "pipe:1",
    ]
    try:
        proc = subprocess.run(cmd, input=pcm16, capture_output=True, check=True)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail="ffmpeg not available in image") from exc
    except subprocess.CalledProcessError as exc:
        logger.error("ffmpeg failed: %s", exc.stderr.decode(errors="replace"))
        raise HTTPException(status_code=500, detail="audio encoding failed")
    return proc.stdout


def _wav_streaming_header(sample_rate: int, channels: int = 1, bits: int = 16) -> bytes:
    """Minimal WAV/RIFF header with 'unknown length' sentinels.

    When streaming, total length is unknown in advance. We emit 0xFFFFFFFF
    in the RIFF and data chunk size fields — most players (ffplay, VLC,
    mpv, web audio) treat that as an open-ended stream and just keep
    reading until EOF. Standard 44-byte PCM header layout.
    """
    byte_rate = sample_rate * channels * bits // 8
    block_align = channels * bits // 8
    max_size = 0xFFFFFFFF
    return (
        b"RIFF"
        + struct.pack("<I", max_size)
        + b"WAVE"
        + b"fmt "
        + struct.pack("<I", 16)
        + struct.pack("<H", 1)
        + struct.pack("<H", channels)
        + struct.pack("<I", sample_rate)
        + struct.pack("<I", byte_rate)
        + struct.pack("<H", block_align)
        + struct.pack("<H", bits)
        + b"data"
        + struct.pack("<I", max_size)
    )


def _chunk_to_pcm16_bytes(chunk) -> bytes:
    """Convert a VoxCPM streaming chunk (float32 numpy) to 16-bit LE PCM."""
    arr = np.asarray(chunk)
    if arr.ndim > 1:
        arr = np.squeeze(arr)
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    arr = np.clip(arr, -1.0, 1.0)
    return (arr * 32767.0).astype("<i2").tobytes()


def _is_upload(value) -> bool:
    return hasattr(value, "filename") and hasattr(value, "read")


def _as_upload_list(form, *keys: str) -> list[UploadFile]:
    uploads: list[UploadFile] = []
    for key in keys:
        for item in form.getlist(key):
            if _is_upload(item):
                uploads.append(item)
    return uploads


async def _persist_upload(upload: UploadFile, target_path: Path) -> Path:
    await upload.seek(0)
    with target_path.open("wb") as fh:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            fh.write(chunk)
    await upload.close()
    return target_path


def _normalize_audio_to_wav(source_path: Path, output_path: Path) -> Path:
    audio, sample_rate = _load_audio_mono(source_path)
    sf.write(str(output_path), audio, sample_rate, format="WAV")
    return output_path


def _load_audio_mono(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    elif audio.ndim > 2:
        audio = np.squeeze(audio)
    if audio.ndim != 1:
        raise HTTPException(status_code=400, detail=f"unsupported audio shape in {path.name}")
    return audio, int(sample_rate)


def _resample_audio(audio: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    if source_sr == target_sr:
        return audio.astype(np.float32)
    if len(audio) == 0:
        return np.zeros(0, dtype=np.float32)
    duration = len(audio) / float(source_sr)
    target_len = max(1, int(round(duration * target_sr)))
    source_positions = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
    target_positions = np.linspace(0.0, 1.0, num=target_len, endpoint=False)
    return np.interp(target_positions, source_positions, audio).astype(np.float32)


def _merge_reference_audios(paths: list[Path], output_path: Path) -> Path:
    if not paths:
        raise HTTPException(status_code=400, detail="at least one reference_audio file is required")

    merged_segments: list[np.ndarray] = []
    target_sr: Optional[int] = None
    for idx, path in enumerate(paths):
        audio, sr = _load_audio_mono(path)
        if target_sr is None:
            target_sr = sr
        else:
            audio = _resample_audio(audio, sr, target_sr)
        merged_segments.append(audio)
        if idx < len(paths) - 1:
            merged_segments.append(np.zeros(int(target_sr * REFERENCE_GAP_MS / 1000), dtype=np.float32))

    assert target_sr is not None
    merged = np.concatenate(merged_segments) if merged_segments else np.zeros(0, dtype=np.float32)
    sf.write(str(output_path), merged, target_sr, format="WAV")
    return output_path


def _normalize_optional_text(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = value.strip()
    return value or None


def _resolve_voice_description(voice: Optional[str], *, use_default_voice: bool) -> Optional[str]:
    voice = _normalize_optional_text(voice)
    if voice is None:
        return VOICE_PRESETS[DEFAULT_VOICE] if use_default_voice else None
    return VOICE_PRESETS.get(voice.lower(), voice)


def _build_generation_text(req: SpeechRequest, *, use_default_voice: bool) -> str:
    clauses: list[str] = []
    description = _resolve_voice_description(req.voice, use_default_voice=use_default_voice)
    if description:
        clauses.append(description)
    style_prompt = _normalize_optional_text(req.style_prompt)
    if style_prompt:
        clauses.append(style_prompt)

    if req.speed <= 0.8:
        clauses.append("slower pace")
    elif req.speed >= 1.25:
        clauses.append("faster pace")

    if clauses:
        return f"({', '.join(clauses)}){req.input}"
    return req.input


async def _parse_speech_request(request: Request) -> tuple[SpeechRequest, Optional[UploadFile], list[UploadFile]]:
    content_type = request.headers.get("content-type", "")
    prompt_audio: Optional[UploadFile] = None
    reference_audios: list[UploadFile] = []

    if "application/json" in content_type:
        payload = await request.json()
    elif "multipart/form-data" in content_type or "application/x-www-form-urlencoded" in content_type:
        form = await request.form()
        payload = {}
        for key in (
            "model",
            "input",
            "voice",
            "response_format",
            "speed",
            "cfg_value",
            "inference_timesteps",
            "style_prompt",
            "prompt_text",
            "stream",
        ):
            value = form.get(key)
            if value is not None and not _is_upload(value):
                payload[key] = value
        prompt_candidate = form.get("prompt_audio")
        if _is_upload(prompt_candidate):
            prompt_audio = prompt_candidate
        reference_audios = _as_upload_list(form, "reference_audio", "reference_audio[]")
    else:
        raise HTTPException(status_code=415, detail="content-type must be application/json or multipart/form-data")

    try:
        parsed = SpeechRequest.model_validate(payload)
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=exc.errors()) from exc

    if prompt_audio is not None and _normalize_optional_text(parsed.prompt_text) is None:
        raise HTTPException(status_code=400, detail="prompt_text is required when prompt_audio is provided")
    if prompt_audio is None and _normalize_optional_text(parsed.prompt_text) is not None:
        raise HTTPException(status_code=400, detail="prompt_audio is required when prompt_text is provided")

    return parsed, prompt_audio, reference_audios


@app.post("/v1/audio/speech")
async def speech(request: Request):
    req, prompt_audio, reference_audios = await _parse_speech_request(request)
    model = get_model()

    cfg_value = req.cfg_value if req.cfg_value is not None else float(os.environ.get("VOXCPM_CFG_VALUE", "2.0"))
    timesteps = (
        req.inference_timesteps
        if req.inference_timesteps is not None
        else int(os.environ.get("VOXCPM_TIMESTEPS", "10"))
    )
    sample_rate = model.tts_model.sample_rate
    fmt = req.response_format.lower()

    tmpdir = tempfile.TemporaryDirectory(prefix="voxcpm-api-")
    tmp_path = Path(tmpdir.name)

    try:
        prompt_wav_path: Optional[str] = None
        if prompt_audio is not None:
            prompt_suffix = Path(prompt_audio.filename or "prompt.audio").suffix or ".audio"
            prompt_upload_path = await _persist_upload(prompt_audio, tmp_path / f"prompt_upload{prompt_suffix}")
            prompt_wav_path = str(_normalize_audio_to_wav(prompt_upload_path, tmp_path / "prompt.wav"))

        reference_paths: list[Path] = []
        for idx, upload in enumerate(reference_audios):
            suffix = Path(upload.filename or f"reference_{idx}.audio").suffix or ".audio"
            saved = await _persist_upload(upload, tmp_path / f"reference_upload_{idx}{suffix}")
            normalized = _normalize_audio_to_wav(saved, tmp_path / f"reference_{idx}.wav")
            reference_paths.append(normalized)

        reference_wav_path: Optional[str] = None
        if reference_paths:
            merged_reference = _merge_reference_audios(reference_paths, tmp_path / "reference_merged.wav")
            reference_wav_path = str(merged_reference)

        generation_text = _build_generation_text(
            req,
            use_default_voice=not bool(reference_wav_path or prompt_wav_path),
        )

        generation_kwargs = {
            "text": generation_text,
            "cfg_value": cfg_value,
            "inference_timesteps": timesteps,
            "prompt_text": _normalize_optional_text(req.prompt_text),
            "prompt_wav_path": prompt_wav_path,
            "reference_wav_path": reference_wav_path,
        }

        # --- Streaming branch -------------------------------------------------
        if req.stream:
            if fmt not in ("pcm", "wav"):
                raise HTTPException(status_code=400, detail=f"streaming only supports pcm or wav, not {fmt}")

            def iter_audio():
                try:
                    if fmt == "wav":
                        yield _wav_streaming_header(sample_rate)
                    for chunk in model.generate_streaming(**generation_kwargs):
                        yield _chunk_to_pcm16_bytes(chunk)
                except Exception:
                    logger.exception("voxcpm streaming failed")
                    return
                finally:
                    tmpdir.cleanup()

            return StreamingResponse(iter_audio(), media_type=CONTENT_TYPES.get(fmt, "application/octet-stream"))

        # --- Buffered branch (default, OpenAI-compatible) --------------------
        try:
            wav = model.generate(**generation_kwargs)
        except Exception:
            logger.exception("voxcpm generate failed")
            raise HTTPException(status_code=500, detail="speech synthesis failed")

        audio_bytes = _encode_audio(np.asarray(wav), sample_rate, req.response_format)
        return Response(content=audio_bytes, media_type=CONTENT_TYPES.get(fmt, "application/octet-stream"))
    finally:
        if not req.stream:
            tmpdir.cleanup()


if __name__ == "__main__":
    port = int(os.environ.get("VOXCPM_PORT", "9100"))
    logger.info("starting voxcpm server on port %s", port)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
