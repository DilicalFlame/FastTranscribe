"""Pre-download local models so the first transcription doesn't wait.

    uv run scripts/download_model.py                 # default model from .env (large-v3-turbo)
    uv run scripts/download_model.py large-v3 hinglish
    uv run scripts/download_model.py all
    uv run scripts/download_model.py --list
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server.config import settings  # noqa: E402
from server.models import PRESETS, download, is_downloaded, model_dir  # noqa: E402


def main(args: list[str]) -> None:
    if "--list" in args:
        for p in PRESETS.values():
            mark = "✓" if is_downloaded(p.key) else " "
            print(f" [{mark}] {p.key:15} ~{p.vram_gb} GB VRAM  {p.label}\n      {p.note}")
        return
    keys = list(PRESETS) if args == ["all"] else (args or [settings.local_model])
    for key in keys:
        if key not in PRESETS:
            sys.exit(f"Unknown model '{key}'. Choose from: {', '.join(PRESETS)} (or 'all').")
        if is_downloaded(key):
            print(f"{key}: already in {model_dir(key)}")
            continue
        download(key)


if __name__ == "__main__":
    main(sys.argv[1:])
