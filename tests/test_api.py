import io
import os

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

os.environ["SKIP_MODEL_LOAD"] = "1"

from app import create_app  # noqa: E402


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
