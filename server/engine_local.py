"""Local transcription with faster-whisper (CTranslate2). Streams segments as they are decoded."""
from __future__ import annotations

import asyncio
import ctypes
import importlib.util
import os
import sys
import threading
import time
from pathlib import Path
from typing import AsyncIterator

import numpy as np

from .config import settings
from .models import PRESETS, ModelPreset, resolve


# ---------------------------------------------------------------- CUDA runtime libraries
def _preload_cuda_libs() -> None:
    """Make pip-installed NVIDIA libs (uv sync --extra cuda) visible to CTranslate2."""
    spec = importlib.util.find_spec("nvidia")
    if not spec or not spec.submodule_search_locations:
        return
    base = Path(list(spec.submodule_search_locations)[0])
    subdir = "bin" if sys.platform == "win32" else "lib"
    dirs = [base / pkg / subdir for pkg in ("cuda_runtime", "cublas", "cudnn", "cuda_nvrtc")]
    dirs = [d for d in dirs if d.is_dir()]
    if sys.platform == "win32":
        for d in dirs:
            os.add_dll_directory(str(d))
            os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")
    else:
        libs = [p for d in dirs for p in sorted(d.glob("lib*.so*"))]
        # dependencies between the libs mean some only load after others: retry a few rounds
        for _ in range(3):
            remaining = []
            for p in libs:
                try:
                    ctypes.CDLL(str(p), mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    remaining.append(p)
            if not remaining or len(remaining) == len(libs):
                break
            libs = remaining


_preload_cuda_libs()

try:
    import ctranslate2
    from faster_whisper import BatchedInferencePipeline, WhisperModel
    from faster_whisper.feature_extractor import FeatureExtractor

    IMPORT_ERROR: str | None = None
except Exception as e:  # pragma: no cover
    IMPORT_ERROR = f"{type(e).__name__}: {e}"


class LocalEngine:
    def __init__(self) -> None:
        self.lock = threading.Lock()          # one model, one job at a time
        self.model = None
        self.pipeline = None
        self.model_key: str | None = None
        self.preset: ModelPreset | None = None
        self.device: str | None = None
        self.compute_type: str | None = None
        self.status = "idle"                  # idle | loading | ready | error
        self.error: str | None = None
        self.note: str | None = None

    # ------------------------------------------------------------ status
    @property
    def available(self) -> bool:
        return settings.local_enabled and IMPORT_ERROR is None

    def describe(self) -> dict:
        from .models import is_downloaded

        cuda = 0
        if IMPORT_ERROR is None:
            try:
                cuda = ctranslate2.get_cuda_device_count()
            except Exception:
                cuda = 0
        return {
            "enabled": settings.local_enabled,
            "available": self.available,
            "import_error": IMPORT_ERROR,
            "status": self.status,
            "error": self.error,
            "note": self.note,
            "model": self.model_key,
            "device": self.device,
            "compute_type": self.compute_type,
            "cuda_devices": cuda,
            "default_model": settings.local_model,
            "presets": [
                {"key": p.key, "label": p.label, "note": p.note, "vram_gb": p.vram_gb,
                 "downloaded": is_downloaded(p.key)}
                for p in PRESETS.values()
            ],
        }

    # ------------------------------------------------------------ loading
    def ensure_loaded(self, key: str) -> None:
        with self.lock:
            self._load_locked(key)

    def _load_locked(self, key: str) -> None:
        if self.model is not None and self.model_key == key:
            return
        self.status, self.error, self.note = "loading", None, None
        try:
            # free the previous model first so two never sit in VRAM together
            self.model = self.pipeline = None
            path, preset = resolve(key)
            model, device, ctype = self._open(path)
            self._fix_mel_bins(model)
            self.model = model
            self.pipeline = BatchedInferencePipeline(model=model)
            self.model_key, self.preset = key, preset
            self.device, self.compute_type = device, ctype
            self.status = "ready"
            print(f"[local] {key} ready on {device} ({ctype})", flush=True)
        except Exception as e:
            self.status, self.error = "error", f"{type(e).__name__}: {e}"
            self.model = self.pipeline = None
            self.model_key = None
            raise

    def _open(self, path: str):
        want = settings.local_device.lower()
        has_cuda = False
        try:
            has_cuda = ctranslate2.get_cuda_device_count() > 0
        except Exception:
            pass
        if want == "cpu":
            devices = ["cpu"]
        elif want == "cuda" or has_cuda:
            devices = ["cuda", "cpu"]
        else:
            devices = ["cpu"]
        last_err: Exception | None = None
        for device in devices:
            if settings.local_compute_type != "auto":
                types = [settings.local_compute_type]
            elif device == "cuda":
                types = ["int8_float16", "int8", "float16", "float32"]  # GTX 10-series lacks fast fp16
            else:
                types = ["int8", "float32"]
            for ctype in types:
                try:
                    model = WhisperModel(
                        path, device=device, compute_type=ctype,
                        cpu_threads=settings.local_cpu_threads,
                        download_root=str(settings.models_dir),
                    )
                    self._warmup(model)  # surfaces missing cuDNN/cuBLAS now, not mid-job
                    if device == "cpu" and want != "cpu":
                        self.note = (
                            f"GPU failed ({last_err}); running on CPU (slow). See README › GPU setup."
                            if "cuda" in devices and last_err else
                            "No NVIDIA GPU detected: running on CPU (slow). See README › GPU setup."
                        )
                    return model, device, ctype
                except Exception as e:
                    last_err = e
                    print(f"[local] could not start on {device}/{ctype}: {e}", flush=True)
        raise RuntimeError(f"Could not load model: {last_err}")

    @staticmethod
    def _warmup(model) -> None:
        noise = (np.random.default_rng(0).standard_normal(16_000) * 0.01).astype(np.float32)
        segs, _ = model.transcribe(noise, beam_size=1, vad_filter=False, language="en")
        for _ in segs:
            pass

    @staticmethod
    def _fix_mel_bins(model) -> None:
        """Some community conversions ship without preprocessor_config.json; large-v3 needs 128 mels."""
        n_mels = getattr(model.model, "n_mels", None)
        if n_mels and model.feature_extractor.mel_filters.shape[0] != n_mels:
            model.feature_extractor = FeatureExtractor(feature_size=n_mels)
            model.frames_per_second = model.feature_extractor.sampling_rate // model.feature_extractor.hop_length

    # ------------------------------------------------------------ transcription
    async def stream(self, audio: np.ndarray, key: str, language: str | None,
                     prompt: str | None, cancel: threading.Event) -> AsyncIterator[dict]:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        put = lambda item: loop.call_soon_threadsafe(queue.put_nowait, item)  # noqa: E731

        def worker() -> None:
            try:
                with self.lock:
                    if self.model is None or self.model_key != key:
                        put({"type": "status", "message": f"Loading model “{key}” (first run downloads it)…"})
                        self._load_locked(key)
                    put({"type": "engine", "engine": "local", "model": key,
                         "device": self.device, "compute_type": self.compute_type, "note": self.note})
                    self._run(audio, language, prompt, put, cancel)
            except Exception as e:
                put({"type": "error", "message": f"{type(e).__name__}: {e}"})
            finally:
                put(None)

        threading.Thread(target=worker, daemon=True).start()
        while True:
            item = await queue.get()
            if item is None:
                break
            yield item

    def _run(self, audio, language, prompt, put, cancel) -> None:
        if self.preset and self.preset.force_language:
            language = self.preset.force_language
        common = dict(
            language=language or None,
            beam_size=max(1, settings.local_beam_size),
            initial_prompt=prompt or None,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500, "speech_pad_ms": 300},
        )
        batch = settings.local_batch_size
        last_end = 0.0
        emitted = False

        def emit(segments, info) -> None:
            nonlocal last_end, emitted
            if not emitted:
                put({"type": "info", "language": info.language,
                     "language_probability": round(float(info.language_probability or 0), 3)})
            for s in segments:
                if cancel.is_set():
                    return
                text = s.text.strip()
                if not text:
                    continue
                emitted = True
                last_end = max(last_end, float(s.end))
                put({"type": "segment", "start": round(float(s.start), 2),
                     "end": round(float(s.end), 2), "text": text})

        if batch > 1:
            try:
                segs, info = self.pipeline.transcribe(audio, batch_size=batch, **common)
                emit(segs, info)
                return
            except Exception as e:
                if "out of memory" not in str(e).lower():
                    raise
                put({"type": "status", "message": "GPU memory full in batched mode; continuing sequentially…"})

        # sequential mode (or OOM fallback: resume where batched mode stopped)
        segs, info = self.model.transcribe(
            audio, condition_on_previous_text=False,
            clip_timestamps=[last_end] if last_end > 0 else "0", **common,
        )
        emit(segs, info)


engine = LocalEngine()


def preload_in_background() -> None:
    if not (engine.available and settings.local_preload):
        return

    def _go():
        t = time.time()
        try:
            engine.ensure_loaded(settings.local_model)
            print(f"[local] preload finished in {time.time() - t:.1f}s", flush=True)
        except Exception as e:
            print(f"[local] preload failed: {e}", flush=True)

    threading.Thread(target=_go, daemon=True).start()
