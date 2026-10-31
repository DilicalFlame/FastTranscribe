"""Check whether the local engine can use your NVIDIA GPU.

    uv run scripts/check_gpu.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server import engine_local as el  # noqa: E402  (also preloads pip CUDA libs)

if el.IMPORT_ERROR:
    sys.exit(f"faster-whisper failed to import: {el.IMPORT_ERROR}")

n = el.ctranslate2.get_cuda_device_count()
print(f"CUDA devices visible to CTranslate2: {n}")
if n == 0:
    sys.exit("No GPU visible. Update the NVIDIA driver, and run:  uv sync --extra cuda")

print("Supported compute types on GPU 0:", sorted(el.ctranslate2.get_supported_compute_types("cuda")))
print("Running a tiny GPU test (downloads the 'small' model once if needed)…")
from server.config import settings  # noqa: E402

settings.local_device = "cuda"
try:
    el.engine.ensure_loaded("small")
    print(f"OK → device={el.engine.device}, compute_type={el.engine.compute_type}")
    if el.engine.device != "cuda":
        print("GPU could not run the model:", el.engine.note)
        print("Fix: uv sync --extra cuda   (installs cuBLAS + cuDNN 9), and update your NVIDIA driver.")
except Exception as e:
    sys.exit(f"Failed: {e}")
