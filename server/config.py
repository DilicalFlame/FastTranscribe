"""Settings, read from environment variables / the .env file in the project root."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def _env(name: str, default: str) -> str:
    value = os.getenv(name, "").strip()
    return value if value else default


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    return _env(name, "true" if default else "false").lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    host: str = _env("HOST", "127.0.0.1")
    port: int = _env_int("PORT", 8000)
    # ---- Integration with an external frontend ----
    # Comma-separated browser origins allowed to call the API, e.g. "http://localhost:5173,https://app.example.com". "*" = any.
    cors_origins: str = _env("CORS_ORIGINS", "*")
    # If set, every /api/* call must send it as "X-API-Key: <key>" (or "Authorization: Bearer <key>").
    api_key: str = _env("API_KEY", "")

    # ---- Local engine (faster-whisper) ----
    local_enabled: bool = _env_bool("LOCAL_ENABLED", True)
    # Default model key (see server/models.py), or a path to a CTranslate2 model folder.
    local_model: str = _env("LOCAL_MODEL", "large-v3-turbo")
    local_device: str = _env("LOCAL_DEVICE", "auto")          # auto | cuda | cpu
    local_compute_type: str = _env("LOCAL_COMPUTE_TYPE", "auto")  # auto | int8_float16 | float16 | int8 | float32
    local_beam_size: int = _env_int("LOCAL_BEAM_SIZE", 5)      # 5 = more accurate on noisy audio; 1 = greedy (fastest)
    local_batch_size: int = _env_int("LOCAL_BATCH_SIZE", 8)    # >1 = batched (fastest on GPU); 0/1 = sequential
    local_cpu_threads: int = _env_int("LOCAL_CPU_THREADS", 0)  # 0 = let CTranslate2 decide
    local_preload: bool = _env_bool("LOCAL_PRELOAD", True)     # load the model when the server starts
    models_dir: Path = Path(_env("MODELS_DIR", str(ROOT / "models")))

    # ---- Online engine (Groq) ----
    groq_api_key: str = _env("GROQ_API_KEY", "")
    groq_model: str = _env("GROQ_MODEL", "whisper-large-v3")  # full v3: better on Hindi/Hinglish than turbo
    groq_base_url: str = _env("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    groq_chunk_seconds: int = _env_int("GROQ_CHUNK_SECONDS", 120)
    groq_parallel: int = _env_int("GROQ_PARALLEL", 3)

    # ---- Audio clean-up for noisy meeting recordings (ffmpeg filters) ----
    enhance_default: bool = _env_bool("ENHANCE_AUDIO", True)
    enhance_filter: str = _env(
        "ENHANCE_FILTER",
        # cut rumble + hiss, light FFT denoise, then even out quiet/loud speakers
        "highpass=f=80,lowpass=f=7800,afftdn=nf=-25,dynaudnorm=f=150:g=15:p=0.9",
    )


settings = Settings()
settings.models_dir.mkdir(parents=True, exist_ok=True)
