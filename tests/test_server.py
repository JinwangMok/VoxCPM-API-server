import importlib.util
import io
import sys
import types
import wave
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient


REPO_ROOT = Path(__file__).resolve().parents[1]
SERVER_PATH = REPO_ROOT / "docker" / "voxcpm" / "server.py"


class DummySoundFileModule:
    @staticmethod
    def write(file_obj, wav, sample_rate, format="WAV"):
        arr = np.asarray(wav, dtype=np.float32)
        if arr.ndim > 1:
            arr = np.squeeze(arr)
        pcm = np.clip(arr, -1.0, 1.0)
        pcm = (pcm * 32767.0).astype("<i2")
        with wave.open(file_obj, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm.tobytes())

    @staticmethod
    def read(path, dtype="float32", always_2d=False):
        with wave.open(str(path), "rb") as wf:
            frames = wf.readframes(wf.getnframes())
            sample_rate = wf.getframerate()
            channels = wf.getnchannels()
        audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32767.0
        if channels > 1:
            audio = audio.reshape(-1, channels)
            if not always_2d:
                audio = audio.mean(axis=1)
        elif always_2d:
            audio = audio.reshape(-1, 1)
        return audio.astype(np.float32), sample_rate


class DummyModel:
    def __init__(self):
        self.tts_model = type("TTS", (), {"sample_rate": 16000})()
        self.calls = []
        self.reference_lengths = []
        self.prompt_lengths = []

    def generate(self, **kwargs):
        self.calls.append(kwargs.copy())
        ref = kwargs.get("reference_wav_path")
        if ref:
            with wave.open(ref, "rb") as wf:
                self.reference_lengths.append(wf.getnframes())
        prompt = kwargs.get("prompt_wav_path")
        if prompt:
            with wave.open(prompt, "rb") as wf:
                self.prompt_lengths.append(wf.getnframes())
        return np.zeros(1600, dtype=np.float32)

    def generate_streaming(self, **kwargs):
        self.calls.append(kwargs.copy())
        yield np.zeros(800, dtype=np.float32)
        yield np.zeros(800, dtype=np.float32)


@pytest.fixture
def server_module(monkeypatch):
    monkeypatch.setitem(sys.modules, "soundfile", DummySoundFileModule())
    spec = importlib.util.spec_from_file_location("voxcpm_server_test", SERVER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    dummy = DummyModel()
    monkeypatch.setattr(module, "get_model", lambda: dummy)
    return module, dummy


@pytest.fixture
def client(server_module):
    module, dummy = server_module
    with TestClient(module.app) as test_client:
        yield test_client, dummy


def _wav_bytes(seconds: float = 0.2, sr: int = 16000, freq: float = 220.0) -> bytes:
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    wav = 0.2 * np.sin(2 * np.pi * freq * t).astype(np.float32)
    pcm = (np.clip(wav, -1.0, 1.0) * 32767.0).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def test_json_speech_request_uses_voice_preset(client):
    test_client, dummy = client

    response = test_client.post(
        "/v1/audio/speech",
        json={
            "model": "voxcpm2",
            "voice": "nova",
            "input": "Hello test.",
            "response_format": "wav",
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert dummy.calls
    assert dummy.calls[0]["reference_wav_path"] is None
    assert dummy.calls[0]["prompt_wav_path"] is None
    assert dummy.calls[0]["text"].startswith("(A young adult female voice, bright and friendly")


def test_multipart_request_supports_multiple_reference_audios_and_prompt_audio(client):
    test_client, dummy = client

    files = [
        ("reference_audio", ("ref1.wav", _wav_bytes(0.2, freq=220.0), "audio/wav")),
        ("reference_audio", ("ref2.wav", _wav_bytes(0.25, freq=330.0), "audio/wav")),
        ("prompt_audio", ("prompt.wav", _wav_bytes(0.15, freq=440.0), "audio/wav")),
    ]
    data = {
        "model": "voxcpm2",
        "input": "Generate with my voice.",
        "prompt_text": "This is the transcript of the prompt audio.",
        "response_format": "wav",
        "style_prompt": "calm, slightly smiling",
    }

    response = test_client.post("/v1/audio/speech", data=data, files=files)

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert len(dummy.calls) == 1
    call = dummy.calls[0]
    assert call["reference_wav_path"]
    assert call["reference_wav_path"].endswith(".wav")
    assert call["prompt_wav_path"]
    assert call["prompt_wav_path"].endswith(".wav")
    assert call["prompt_text"] == "This is the transcript of the prompt audio."
    assert "calm, slightly smiling" in call["text"]
    assert dummy.reference_lengths[0] > 6000
    assert dummy.prompt_lengths[0] > 2000


def test_single_reference_audio_is_also_normalized_to_wav(client):
    test_client, dummy = client

    files = [
        ("reference_audio", ("ref1.wav", _wav_bytes(0.2, freq=220.0), "audio/wav")),
    ]
    data = {
        "model": "voxcpm2",
        "input": "Generate with my voice.",
        "response_format": "wav",
    }

    response = test_client.post("/v1/audio/speech", data=data, files=files)

    assert response.status_code == 200
    assert dummy.calls[0]["reference_wav_path"].endswith(".wav")


def test_prompt_audio_requires_prompt_text(client):
    test_client, _dummy = client

    files = [
        ("prompt_audio", ("prompt.wav", _wav_bytes(0.15, freq=440.0), "audio/wav")),
    ]
    data = {
        "model": "voxcpm2",
        "input": "Generate with my voice.",
        "response_format": "wav",
    }

    response = test_client.post("/v1/audio/speech", data=data, files=files)

    assert response.status_code == 400
    assert "prompt_text" in response.json()["detail"]


def test_json_request_accepts_custom_voice_description(client):
    test_client, dummy = client

    response = test_client.post(
        "/v1/audio/speech",
        json={
            "model": "voxcpm2",
            "voice": "A warm middle-aged male voice, reflective and gentle",
            "input": "Hello test.",
            "response_format": "wav",
        },
    )

    assert response.status_code == 200
    assert dummy.calls[0]["text"].startswith("(A warm middle-aged male voice, reflective and gentle)")
