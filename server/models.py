"""Local model presets. Each one is a CTranslate2 (faster-whisper) model on Hugging Face."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import settings


@dataclass(frozen=True)
class ModelPreset:
    key: str
    repo: str
    label: str
    vram_gb: float          # rough int8 VRAM need
    force_language: str | None = None   # some fine-tunes need a fixed language token
    note: str = ""


PRESETS: dict[str, ModelPreset] = {
    p.key: p
    for p in [
        ModelPreset(
            "large-v3-turbo",
            "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
            "Whisper large-v3-turbo · fast, 99 languages",
            1.6,
            note="Best speed/accuracy balance. Default.",
        ),
        ModelPreset(
            "large-v3",
            "Systran/faster-whisper-large-v3",
            "Whisper large-v3 · most accurate, ~3x slower",
            3.0,
            note="Better on Hindi and noisy audio than turbo. Fits 4 GB in int8.",
        ),
        ModelPreset(
            "hinglish",
            "Hub84/faster-whisper-hinglish-prime",
            "Hinglish Prime · Hindi/Hinglish in Roman script",
            1.6,
            force_language="en",  # this fine-tune emits Roman-script Hinglish under the 'en' token
            note="Community CT2 conversion of Oriserve/Whisper-Hindi2Hinglish-Prime. Experimental.",
        ),
        ModelPreset(
            "small",
            "Systran/faster-whisper-small",
            "Whisper small · for CPU-only machines",
            0.6,
            note="Much lower accuracy; use only when there is no GPU.",
        ),
    ]
}


def model_dir(key: str) -> Path:
    return settings.models_dir / key


def is_downloaded(key: str) -> bool:
    d = model_dir(key)
    return (d / "model.bin").is_file()


def resolve(key_or_path: str) -> tuple[str, ModelPreset | None]:
    """Return (path_or_repo_to_load, preset). Downloads the preset into ./models if needed."""
    if key_or_path in PRESETS:
        preset = PRESETS[key_or_path]
        if not is_downloaded(preset.key):
            download(preset.key)
        return str(model_dir(preset.key)), preset
    # A custom folder path or a raw HF repo id
    return key_or_path, None


def download(key: str) -> Path:
    from huggingface_hub import snapshot_download

    preset = PRESETS[key]
    target = model_dir(key)
    print(f"[models] downloading {preset.repo} -> {target} (one-time)…", flush=True)
    try:
        snapshot_download(repo_id=preset.repo, local_dir=str(target))
    except Exception as e:
        raise RuntimeError(
            f"Could not download model '{key}' from Hugging Face ({type(e).__name__}: {e}). "
            "Check your internet connection, or see README › 'Model download blocked'."
        ) from e
    print(f"[models] {key} ready.", flush=True)
    return target
