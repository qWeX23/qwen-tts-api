import io
import os

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

os.environ["SKIP_MODEL_LOAD"] = "1"

from app import create_app, parse_bool_env, select_dtype  # noqa: E402


client = TestClient(create_app())


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True


def test_custom_missing_speaker():
    response = client.post("/tts/custom", json={"text": "hi"})
    assert response.status_code == 400


def test_clone_requires_exact_ref():
    response = client.post(
        "/tts/clone",
        json={"text": "hello", "ref_audio_url": "https://example.com/a.wav", "ref_audio_base64": "Zg=="},
    )
    assert response.status_code == 400


def test_custom_audio_response():
    response = client.post(
        "/tts/custom",
        json={"text": "hello", "speaker": "dummy"},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("audio/wav")
    audio, sample_rate = sf.read(io.BytesIO(response.content))
    assert isinstance(audio, np.ndarray)
    assert sample_rate == 24000


class _FakeCuda:
    def __init__(self, capability):
        self._capability = capability

    def is_available(self):
        return True

    def is_bf16_supported(self):
        return True

    def get_device_capability(self, _index):
        return self._capability


class _FakeTorch:
    float32 = "float32"
    float16 = "float16"
    bfloat16 = "bfloat16"

    def __init__(self, capability):
        self.cuda = _FakeCuda(capability)


def test_select_dtype_auto_uses_fp16_for_pre_ampere(monkeypatch):
    monkeypatch.setattr("app.get_torch", lambda: _FakeTorch((7, 5)))
    dtype, label = select_dtype("auto", "cuda:0")
    assert dtype == "float16"
    assert label == "fp16"


def test_select_dtype_auto_uses_bf16_for_ampere_plus(monkeypatch):
    monkeypatch.setattr("app.get_torch", lambda: _FakeTorch((8, 0)))
    dtype, label = select_dtype("auto", "cuda:0")
    assert dtype == "bfloat16"
    assert label == "bf16"


def test_parse_bool_env_true(monkeypatch):
    monkeypatch.setenv("TEST_BOOL_FLAG", "yes")
    assert parse_bool_env("TEST_BOOL_FLAG") is True


def test_parse_bool_env_default_false(monkeypatch):
    monkeypatch.delenv("TEST_BOOL_FLAG", raising=False)
    assert parse_bool_env("TEST_BOOL_FLAG") is False
