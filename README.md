# Qwen3-TTS REST API

Native REST service for Qwen3-TTS with CPU, NVIDIA CUDA, and AMD ROCm Docker targets.

## Quickstart

### CPU

```bash
docker compose --profile cpu up --build
```

### NVIDIA (CUDA)

```bash
docker compose --profile nvidia up --build
```

Requires `nvidia-container-toolkit` and `--gpus all` support.
The NVIDIA profile defaults to `DTYPE=fp16` for better throughput on pre-Ampere GPUs (for example RTX 20-series).
It also enables startup warmup so the first external TTS request is fast after the container reports healthy.

### AMD (ROCm)

```bash
docker compose --profile amd up --build
```

Requires `/dev/kfd` and `/dev/dri` access on the host and a supported ROCm stack.

### Hugging Face cache

Mount the Hugging Face cache to avoid repeated downloads:

```bash
mkdir -p ~/.cache/huggingface
docker compose --profile cpu up --build
```

## API

Base URL: `http://localhost:8000`

### Health

`GET /health`

### Voices

`GET /voices`

### Custom voice

`POST /tts/custom`

```bash
curl -X POST http://localhost:8000/tts/custom \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: your-key' \
  -d '{"text":"Hello","language":"Auto","speaker":"Alice","instruct":"friendly"}' \
  --output tts.wav
```

### Voice design

`POST /tts/design`

```bash
curl -X POST http://localhost:8000/tts/design \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: your-key' \
  -d '{"text":"Hello","language":"Auto","instruct":"warm and calm"}' \
  --output tts.wav
```

### Voice clone

`POST /tts/clone`

```bash
curl -X POST http://localhost:8000/tts/clone \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: your-key' \
  -d '{"text":"Hello","ref_audio_url":"https://example.com/ref.wav"}' \
  --output tts.wav
```

## Configuration

| Env var | Default | Description |
| --- | --- | --- |
| `DEVICE` | `auto` | `auto`, `cpu`, or `cuda:0` |
| `DTYPE` | `auto` | `auto`, `fp32`, `fp16`, `bf16` |
| `ATTN_IMPL` | `sdpa` | `sdpa`, `flash_attention_2`, or `auto` |
| `MODEL_CUSTOM` | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | Custom voice model ID |
| `MODEL_DESIGN` | `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` | Voice design model ID |
| `MODEL_BASE` | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | Base model ID |
| `WARMUP_CUSTOM_ON_STARTUP` | `0` | Run one startup warmup TTS request using the custom voice model |
| `WARMUP_TEXT` | `Warmup request` | Warmup text used when startup warmup is enabled |
| `WARMUP_SPEAKER` | unset | Specific speaker for warmup; defaults to the first available speaker |
| `API_KEY` | unset | Require `X-API-Key` header when set |
| `MAX_TEXT_CHARS` | `2000` | Max input characters |
| `MAX_CONCURRENT_JOBS` | `1` (GPU) / `2` (CPU) | Limit concurrent inferences |
| `MAX_REQUEST_BYTES` | `10485760` | Max request body size |

## Troubleshooting

- **No GPU detected**: verify `nvidia-smi` or `rocminfo` works on the host.
- **Slow NVIDIA inference**: keep `DTYPE=fp16` for RTX 20-series cards, send a warm-up request after startup, and verify `GET /health` reports `device: cuda:0`.
- **Warmup takes too long at startup**: this is expected on cold caches because model files are downloaded and initialized. Subsequent container starts and requests are much faster.
- **Model download failures**: ensure the container can reach Hugging Face and cache volume is writable.
- **ROCm errors**: confirm `/dev/kfd` and `/dev/dri` are passed through and the host ROCm stack matches your GPU.
- **Missing `qwen_tts` dependencies**: ensure the container installs `qwen-tts` and system packages like `sox`/`libgomp1` (rebuild the image after changes). Use `SKIP_MODEL_LOAD=1` only for local testing.

## Development

Install dependencies:

```bash
pip install -r requirements.txt -r requirements-dev.txt
```

Install an appropriate PyTorch build for your hardware (CPU/CUDA/ROCm) before running locally. We recommend installing torch before the rest of the requirements to avoid pip resolving an incompatible wheel during `qwen-tts` installation.

```bash
pip install torch==2.3.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt -r requirements-dev.txt
```

Run tests:

```bash
SKIP_MODEL_LOAD=1 pytest
```
