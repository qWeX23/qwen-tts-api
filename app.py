import asyncio
import base64
import importlib
import importlib.util
import io
import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import requests
import soundfile as sf
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, field_validator
from starlette.requests import Request as StarletteRequest
from starlette.responses import JSONResponse
from starlette.status import HTTP_400_BAD_REQUEST
from starlette.concurrency import run_in_threadpool

logger = logging.getLogger("qwen_tts_api")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))


@dataclass
class Settings:
    device: str
    dtype: Any
    dtype_label: str
    attn_impl: str
    model_custom: str
    model_design: str
    model_base: str
    api_key: Optional[str]
    max_text_chars: int
    max_request_bytes: int
    max_concurrent_jobs: int
    app_version: str


class CustomRequest(BaseModel):
    text: str = Field(..., min_length=1)
    language: str = "Auto"
    speaker: str = Field(..., min_length=1)
    instruct: Optional[str] = None
    format: str = "wav"


class DesignRequest(BaseModel):
    text: str = Field(..., min_length=1)
    language: str = "Auto"
    instruct: str = Field(..., min_length=1)
    format: str = "wav"


class CloneRequest(BaseModel):
    text: str = Field(..., min_length=1)
    language: str = "Auto"
    ref_audio_url: Optional[str] = None
    ref_audio_base64: Optional[str] = None
    ref_text: Optional[str] = None
    x_vector_only_mode: bool = False
    format: str = "wav"

    @field_validator("ref_audio_url")
    @classmethod
    def validate_ref_audio_url(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        if not value.startswith("http://") and not value.startswith("https://"):
            raise ValueError("ref_audio_url must be http(s)")
        return value


class DummyTTS:
    def list_speakers(self) -> Tuple[list[str], list[str]]:
        return ["dummy"], ["Auto"]

    def synthesize_custom(self, text: str, language: str, speaker: str, instruct: Optional[str]) -> Tuple[np.ndarray, int]:
        return self._silence()

    def synthesize_design(self, text: str, language: str, instruct: str) -> Tuple[np.ndarray, int]:
        return self._silence()

    def synthesize_clone(
        self,
        text: str,
        language: str,
        ref_audio: bytes,
        ref_text: Optional[str],
        x_vector_only_mode: bool,
    ) -> Tuple[np.ndarray, int]:
        return self._silence()

    def _silence(self) -> Tuple[np.ndarray, int]:
        sample_rate = 24000
        audio = np.zeros(sample_rate, dtype=np.float32)
        return audio, sample_rate


class ModelManager:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.custom_model: Any = None
        self.design_model: Any = None
        self.base_model: Any = None
        self._custom_loaded = False
        self._design_loaded = False
        self._base_loaded = False
        self._custom_lock = threading.Lock()
        self._design_lock = threading.Lock()
        self._base_lock = threading.Lock()
        
        if os.getenv("SKIP_MODEL_LOAD") == "1":
            logger.warning("SKIP_MODEL_LOAD enabled; using dummy TTS model")
            dummy = DummyTTS()
            self.custom_model = dummy
            self.design_model = dummy
            self.base_model = dummy
            self._custom_loaded = True
            self._design_loaded = True
            self._base_loaded = True
    
    def _get_model_class(self):
        """Lazy import to avoid loading at startup"""
        try:
            from qwen_tts import Qwen3TTSModel  # type: ignore
            return Qwen3TTSModel
        except ImportError as exc:
            raise RuntimeError(
                "Failed to import qwen_tts. Ensure the Qwen3-TTS package and its "
                "dependencies are installed (pip install qwen-tts) and required "
                "system libraries like sox/libgomp1 are available. Original error: "
                f"{exc}"
            ) from exc
    
    def _load_custom_model(self) -> Any:
        """Lazy load custom voice model"""
        if self._custom_loaded:
            return self.custom_model
        
        with self._custom_lock:
            if self._custom_loaded:
                return self.custom_model
            
            logger.info("Lazy loading custom voice model: %s", self.settings.model_custom)
            start = time.time()
            Qwen3TTSModel = self._get_model_class()
            self.custom_model = Qwen3TTSModel.from_pretrained(
                self.settings.model_custom,
                device_map=self.settings.device,
                torch_dtype=self.settings.dtype,
                attn_implementation=self.settings.attn_impl,
            )
            self._custom_loaded = True
            logger.info("Custom voice model loaded in %.2f seconds", time.time() - start)
            return self.custom_model
    
    def _load_design_model(self) -> Any:
        """Lazy load voice design model"""
        if self._design_loaded:
            return self.design_model
        
        with self._design_lock:
            if self._design_loaded:
                return self.design_model
            
            logger.info("Lazy loading voice design model: %s", self.settings.model_design)
            start = time.time()
            Qwen3TTSModel = self._get_model_class()
            self.design_model = Qwen3TTSModel.from_pretrained(
                self.settings.model_design,
                device_map=self.settings.device,
                torch_dtype=self.settings.dtype,
                attn_implementation=self.settings.attn_impl,
            )
            self._design_loaded = True
            logger.info("Voice design model loaded in %.2f seconds", time.time() - start)
            return self.design_model
    
    def _load_base_model(self) -> Any:
        """Lazy load base/clone model"""
        if self._base_loaded:
            return self.base_model
        
        with self._base_lock:
            if self._base_loaded:
                return self.base_model
            
            logger.info("Lazy loading base model: %s", self.settings.model_base)
            start = time.time()
            Qwen3TTSModel = self._get_model_class()
            self.base_model = Qwen3TTSModel.from_pretrained(
                self.settings.model_base,
                device_map=self.settings.device,
                torch_dtype=self.settings.dtype,
                attn_implementation=self.settings.attn_impl,
            )
            self._base_loaded = True
            logger.info("Base model loaded in %.2f seconds", time.time() - start)
            return self.base_model

    def list_voices(self) -> Tuple[list[str], list[str], Optional[str]]:
        warning = None
        
        # Try to load custom model if not loaded
        try:
            model = self._load_custom_model()
        except Exception as exc:
            logger.warning("Failed to load custom model for voice listing: %s", exc)
            return [], [], f"custom model not loaded: {exc}"
        
        if model is None:
            return [], [], "custom model not loaded"

        for attr in ("list_speakers", "speakers", "speaker_list"):
            if hasattr(model, attr):
                speakers = getattr(model, attr)
                if callable(speakers):
                    speakers = speakers()
                languages = []
                if isinstance(speakers, tuple) and len(speakers) == 2:
                    speakers, languages = speakers
                if not languages and hasattr(model, "languages"):
                    languages = list(getattr(model, "languages"))
                return list(speakers), list(languages), None
        warning = "speakers could not be enumerated"
        return [], [], warning

    def synthesize_custom(self, payload: CustomRequest) -> Tuple[np.ndarray, int]:
        model = self._load_custom_model()
        if hasattr(model, "generate_custom_voice"):
            return model.generate_custom_voice(
                text=payload.text,
                language=payload.language,
                speaker=payload.speaker,
                instruct=payload.instruct
            )
        raise RuntimeError("Custom voice model does not support synthesis")

    def synthesize_design(self, payload: DesignRequest) -> Tuple[np.ndarray, int]:
        model = self._load_design_model()
        if hasattr(model, "generate_voice_design"):
            return model.generate_voice_design(
                text=payload.text,
                language=payload.language,
                instruct=payload.instruct
            )
        raise RuntimeError("Voice design model does not support synthesis")

    def synthesize_clone(
        self, payload: CloneRequest, ref_audio: bytes
    ) -> Tuple[np.ndarray, int]:
        model = self._load_base_model()
        if hasattr(model, "generate_voice_clone"):
            return model.generate_voice_clone(
                text=payload.text,
                language=payload.language,
                ref_audio=ref_audio,
                ref_text=payload.ref_text,
                x_vector_only_mode=payload.x_vector_only_mode,
            )
        raise RuntimeError("Voice clone model does not support synthesis")


def select_device(device_env: str) -> str:
    torch_module = get_torch()
    if device_env != "auto":
        return device_env
    if torch_module and torch_module.cuda.is_available():
        return "cuda:0"
    return "cpu"


def select_dtype(dtype_env: str, device: str) -> Tuple[Any, str]:
    torch_module = get_torch()
    if torch_module is None:
        if dtype_env not in {"auto", "fp32"}:
            raise ValueError("torch is required for fp16/bf16 selection")
        return None, "fp32"
    if dtype_env != "auto":
        mapping = {
            "fp32": torch_module.float32,
            "fp16": torch_module.float16,
            "bf16": torch_module.bfloat16,
        }
        if dtype_env not in mapping:
            raise ValueError(f"Unsupported dtype: {dtype_env}")
        return mapping[dtype_env], dtype_env
    if device.startswith("cuda") and torch_module.cuda.is_available():
        if torch_module.cuda.is_bf16_supported():
            return torch_module.bfloat16, "bf16"
        return torch_module.float16, "fp16"
    return torch_module.float32, "fp32"


def get_torch() -> Optional[Any]:
    if importlib.util.find_spec("torch") is None:
        return None
    return importlib.import_module("torch")


def parse_settings() -> Settings:
    torch_module = get_torch()
    if torch_module is None and os.getenv("SKIP_MODEL_LOAD") != "1":
        raise RuntimeError("torch is required to run the API. Install torch or set SKIP_MODEL_LOAD=1.")
    device_env = os.getenv("DEVICE", "auto")
    device = select_device(device_env)
    dtype_env = os.getenv("DTYPE", "auto")
    dtype, dtype_label = select_dtype(dtype_env, device)
    max_text_chars = int(os.getenv("MAX_TEXT_CHARS", "2000"))
    max_request_bytes = int(os.getenv("MAX_REQUEST_BYTES", str(10 * 1024 * 1024)))
    max_concurrent_default = 2 if device == "cpu" else 1
    max_concurrent_jobs = int(os.getenv("MAX_CONCURRENT_JOBS", str(max_concurrent_default)))
    return Settings(
        device=device,
        dtype=dtype,
        dtype_label=dtype_label,
        attn_impl=os.getenv("ATTN_IMPL", "sdpa"),
        model_custom=os.getenv("MODEL_CUSTOM", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"),
        model_design=os.getenv("MODEL_DESIGN", "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"),
        model_base=os.getenv("MODEL_BASE", "Qwen/Qwen3-TTS-12Hz-1.7B-Base"),
        api_key=os.getenv("API_KEY"),
        max_text_chars=max_text_chars,
        max_request_bytes=max_request_bytes,
        max_concurrent_jobs=max_concurrent_jobs,
        app_version=os.getenv("APP_VERSION", "0.1.0"),
    )


def audio_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format="WAV")
    return buffer.getvalue()


def normalize_audio_output(output: Any) -> Tuple[np.ndarray, int]:
    # Qwen3TTS returns (audio_list, sample_rate) where audio_list is a list of numpy arrays
    if isinstance(output, tuple) and len(output) == 2:
        audio_data, sample_rate = output
        # If audio_data is a list (batch), take the first item
        if isinstance(audio_data, list) and len(audio_data) > 0:
            audio_data = audio_data[0]
        return np.asarray(audio_data), int(sample_rate)
    if isinstance(output, dict):
        audio = output.get("audio") or output.get("waveform")
        sample_rate = output.get("sampling_rate") or output.get("sample_rate")
        if audio is not None and sample_rate is not None:
            return np.asarray(audio), int(sample_rate)
    if isinstance(output, np.ndarray):
        return output, 24000
    if isinstance(output, list) and len(output) > 0:
        # Handle list of audio arrays
        return np.asarray(output[0]), 24000
    raise RuntimeError(f"Unsupported audio output format from model: {type(output)}")


def create_app() -> FastAPI:
    settings = parse_settings()
    manager = ModelManager(settings)
    semaphore = asyncio.Semaphore(settings.max_concurrent_jobs)

    app = FastAPI(title="Qwen3-TTS API", version=settings.app_version)
    app.state.settings = settings
    app.state.manager = manager
    app.state.semaphore = semaphore

    # Mount static files
    import os as os_module
    static_dir = os_module.path.join(os_module.path.dirname(__file__), "static")
    if os_module.path.exists(static_dir):
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/")
    async def root():
        index_path = os_module.path.join(static_dir, "index.html")
        if os_module.path.exists(index_path):
            return FileResponse(index_path)
        return {"message": "Qwen3-TTS API", "docs": "/docs", "health": "/health"}

    @app.middleware("http")
    async def enforce_api_key(request: Request, call_next):
        if settings.api_key:
            provided = request.headers.get("X-API-Key")
            if provided != settings.api_key:
                return Response(status_code=401, content="Unauthorized")
        return await call_next(request)

    @app.middleware("http")
    async def enforce_max_body(request: Request, call_next):
        if request.method in {"POST", "PUT", "PATCH"}:
            content_length = request.headers.get("content-length")
            if content_length and int(content_length) > settings.max_request_bytes:
                return Response(status_code=413, content="Request body too large")
        return await call_next(request)

    def validate_text(text: str) -> None:
        if len(text) > settings.max_text_chars:
            raise HTTPException(status_code=400, detail="text exceeds MAX_TEXT_CHARS")

    async def run_job(func, *args):
        try:
            await asyncio.wait_for(semaphore.acquire(), timeout=0.001)
        except asyncio.TimeoutError as exc:
            raise HTTPException(status_code=429, detail="Server busy") from exc
        start = time.time()
        try:
            output = await run_in_threadpool(func, *args)
            return output, time.time() - start
        finally:
            semaphore.release()

    @app.get("/health")
    async def health() -> Dict[str, Any]:
        gpu_name = None
        torch_module = get_torch()
        if torch_module and torch_module.cuda.is_available():
            gpu_name = torch_module.cuda.get_device_name(0)
        
        # Check which models are loaded
        manager = app.state.manager
        models_status = {
            "custom_loaded": manager._custom_loaded,
            "design_loaded": manager._design_loaded,
            "base_loaded": manager._base_loaded,
        }
        
        return {
            "ok": True,
            "device": settings.device,
            "dtype": settings.dtype_label,
            "models_loaded": any([manager._custom_loaded, manager._design_loaded, manager._base_loaded]),
            "models_status": models_status,
            "lazy_loading": True,
            "version": settings.app_version,
            "gpu": gpu_name,
        }

    @app.get("/voices")
    async def voices() -> Dict[str, Any]:
        speakers, languages, warning = app.state.manager.list_voices()
        payload: Dict[str, Any] = {"speakers": speakers, "languages": languages}
        if warning:
            payload["warning"] = warning
        return payload

    @app.post("/tts/custom")
    async def tts_custom(payload: CustomRequest) -> Response:
        validate_text(payload.text)
        if payload.format.lower() != "wav":
            raise HTTPException(status_code=400, detail="Only wav format is supported")

        output, duration = await run_job(app.state.manager.synthesize_custom, payload)
        audio, sample_rate = normalize_audio_output(output)
        wav_bytes = audio_to_wav_bytes(audio, sample_rate)
        logger.info(
            "tts_custom duration_ms=%s device=%s success=true",
            int(duration * 1000),
            settings.device,
        )
        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=tts.wav"},
        )

    @app.post("/tts/design")
    async def tts_design(payload: DesignRequest) -> Response:
        validate_text(payload.text)
        if payload.format.lower() != "wav":
            raise HTTPException(status_code=400, detail="Only wav format is supported")

        output, duration = await run_job(app.state.manager.synthesize_design, payload)
        audio, sample_rate = normalize_audio_output(output)
        wav_bytes = audio_to_wav_bytes(audio, sample_rate)
        logger.info(
            "tts_design duration_ms=%s device=%s success=true",
            int(duration * 1000),
            settings.device,
        )
        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=tts.wav"},
        )

    @app.post("/tts/clone")
    async def tts_clone(payload: CloneRequest) -> Response:
        validate_text(payload.text)
        if payload.format.lower() != "wav":
            raise HTTPException(status_code=400, detail="Only wav format is supported")

        if bool(payload.ref_audio_url) == bool(payload.ref_audio_base64):
            raise HTTPException(
                status_code=400,
                detail="Provide exactly one of ref_audio_url or ref_audio_base64",
            )

        ref_audio_bytes = await run_in_threadpool(load_reference_audio, payload)
        output, duration = await run_job(app.state.manager.synthesize_clone, payload, ref_audio_bytes)
        audio, sample_rate = normalize_audio_output(output)
        wav_bytes = audio_to_wav_bytes(audio, sample_rate)
        logger.info(
            "tts_clone duration_ms=%s device=%s success=true",
            int(duration * 1000),
            settings.device,
        )
        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=tts.wav"},
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(request: StarletteRequest, exc: RequestValidationError):
        return JSONResponse(status_code=HTTP_400_BAD_REQUEST, content={"detail": exc.errors()})

    @app.exception_handler(Exception)
    async def handle_exception(request: Request, exc: Exception):
        logger.exception("Unhandled exception on %s", request.url.path)
        return Response(status_code=500, content=str(exc))

    return app


def load_reference_audio(payload: CloneRequest) -> bytes:
    if payload.ref_audio_url:
        response = requests.get(payload.ref_audio_url, timeout=20)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to fetch ref_audio_url")
        return response.content
    if payload.ref_audio_base64:
        try:
            return base64.b64decode(payload.ref_audio_base64)
        except base64.binascii.Error as exc:
            raise HTTPException(status_code=400, detail="Invalid base64 ref audio") from exc
    raise HTTPException(status_code=400, detail="ref audio required")


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
