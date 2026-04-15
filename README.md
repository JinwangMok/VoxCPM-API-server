# VoxCPM API Server

OpenAI-compatible TTS HTTP server that wraps
[OpenBMB/VoxCPM2](https://github.com/OpenBMB/VoxCPM) in a single container.
Designed to drop into the
[dgx-spark-ai-cluster](https://github.com/JinwangMok/dgx-spark-ai-cluster)
layout (aarch64 Ubuntu 24.04, CUDA, Docker Compose) next to the existing
`vllm` and `whisper` services.

## Layout

```
docker/voxcpm/Dockerfile      container image
docker/voxcpm/server.py       FastAPI app exposing OpenAI /v1/audio/speech
docker/voxcpm/requirements.txt
scripts/build-and-push.sh     buildx → Docker Hub (jinwangmok/voxcpm-api-server)
tests/e2e.sh                  end-to-end check against a running instance
```

## OpenAI compatibility

| Endpoint | Status | Notes |
|---|---|---|
| `POST /v1/audio/speech` | ✅ | JSON mode: `model`, `input`, `voice`, `response_format`, `speed` |
| `POST /v1/audio/speech` | ✅ | Multipart mode: upload reference/prompt audio for cloning |
| `GET  /v1/models`       | ✅ | returns `voxcpm2` |
| `GET  /health`          | ✅ | non-OpenAI liveness probe |

Voice names `alloy / echo / fable / onyx / nova / shimmer` are mapped to
VoxCPM2 Voice Design prompts (natural-language descriptions), so existing
OpenAI clients work without modification. `speed` is translated to VoxCPM
pace steering (`slower pace` / `faster pace`).

Response formats: `wav`, `flac`, `mp3`, `opus`, `aac`, `pcm`.
`mp3/opus/aac` are encoded via ffmpeg inside the container.

VoxCPM-specific knobs are exposed as optional extra fields:

- `cfg_value` (float, default 2.0)
- `inference_timesteps` (int, default 10)
- `style_prompt` (string, optional natural-language steering)
- `prompt_text` (string, required when `prompt_audio` is supplied)
- `stream` (bool, default false; only `wav` / `pcm`)

### JSON mode (OpenAI-compatible preset voices)

```bash
curl -X POST http://localhost:9100/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -o out.wav \
  -d '{
    "model": "voxcpm2",
    "voice": "nova",
    "input": "Hello from the VoxCPM API server.",
    "response_format": "wav"
  }'
```

### JSON mode with custom voice/style description

The `voice` field may also be a raw natural-language description.

```bash
curl -X POST http://localhost:9100/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -o out.wav \
  -d '{
    "model": "voxcpm2",
    "voice": "A warm middle-aged male voice, reflective and gentle",
    "input": "This uses VoxCPM voice design with a custom description.",
    "response_format": "wav"
  }'
```

### Multipart mode for reference-audio cloning

Use multipart form-data when you want to upload one or more reference clips.
Repeated `reference_audio` parts are accepted; the server merges them into a
single temporary reference WAV with short silence gaps. `prompt_audio` +
`prompt_text` enables the higher-fidelity continuation-style cloning path.

```bash
curl -X POST http://localhost:9100/v1/audio/speech \
  -o styled.wav \
  -F model=voxcpm2 \
  -F 'input=이 문장을 내 목소리 스타일로 읽어줘.' \
  -F 'style_prompt=calm, slightly smiling, conversational' \
  -F response_format=wav \
  -F reference_audio=@samples/ref1.wav \
  -F reference_audio=@samples/ref2.wav \
  -F prompt_audio=@samples/prompt.wav \
  -F 'prompt_text=프롬프트 오디오의 정확한 전사 텍스트'
```

Notes:

- If `prompt_audio` is provided, `prompt_text` is required.
- If you omit `voice` during multipart cloning, the uploaded reference audio
  drives the timbre and the optional `style_prompt` steers expression.
- If you include `voice` during multipart cloning, it can be either an OpenAI
  preset name or a free-form description layered on top of the cloned voice.

## Environment

| Var | Default | Purpose |
|---|---|---|
| `VOXCPM_MODEL` | `openbmb/VoxCPM2` | HF repo id or local path |
| `VOXCPM_PORT`  | `9100` | listen port |
| `VOXCPM_CACHE_DIR` | `/data/models/voxcpm` | weight cache |
| `HF_HOME` | `/data/models` | HuggingFace cache root |
| `VOXCPM_CFG_VALUE` | `2.0` | default CFG |
| `VOXCPM_TIMESTEPS` | `10` | default inference steps |

## Build & push

```bash
# One-time QEMU registration if building arm64 on an x86 host
docker run --privileged --rm tonistiigi/binfmt --install arm64

# Build + push to Docker Hub
DOCKER_HUB_USER=jinwangmok ./scripts/build-and-push.sh
```

Image tag: `jinwangmok/voxcpm-api-server:latest` (linux/arm64 by default).

## Run locally on a GPU host

```bash
docker run --rm --gpus all \
  -p 9100:9100 \
  -v /data/models/voxcpm:/data/models/voxcpm \
  jinwangmok/voxcpm-api-server:latest
```

First boot downloads the ~2B weights from HuggingFace into the mounted
cache directory.

## End-to-end test

```bash
BASE_URL=http://<gpu-host>:9100 ./tests/e2e.sh
```

Checks `/health`, `/v1/models`, and `POST /v1/audio/speech` with three
voice presets; validates RIFF/WAVE headers and non-trivial size.

## Integration with dgx-spark-ai-cluster

Add the service next to `vllm` and `whisper` in
`docker/docker-compose.node.yml` and `docker-compose.single.yml`, and add
a `/tts/` upstream to `config/nginx.conf.template` / `nginx-single.conf`.
Suggested port: `9100` (vLLM 8000, whisper 9000, voxcpm 9100).
