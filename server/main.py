"""FastAPI app: serves the SPA and streams transcripts as NDJSON (one JSON object per line)."""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import engine_groq
from .audio import SAMPLE_RATE, load_audio
from .config import ROOT, settings
from .engine_local import engine as local_engine
from .engine_local import preload_in_background
from .models import PRESETS

STATIC = ROOT / "static"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    preload_in_background()
    yield


app = FastAPI(title="FastTranscribe", lifespan=lifespan)


@app.middleware("http")
async def require_api_key(request: Request, call_next):
    if settings.api_key and request.url.path.startswith("/api/") and request.method != "OPTIONS":
        given = request.headers.get("x-api-key") or request.headers.get("authorization", "").removeprefix("Bearer ").strip()
        if given != settings.api_key:
            return JSONResponse({"error": "invalid or missing API key"}, status_code=401)
    return await call_next(request)


# Added after the auth middleware so it is outermost: 401s and preflights also get CORS headers.
_origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
app.add_middleware(CORSMiddleware, allow_origins=_origins, allow_methods=["*"], allow_headers=["*"])

app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/status")
def status():
    return {
        "local": local_engine.describe(),
        "groq": {"configured": bool(settings.groq_api_key), "model": settings.groq_model},
        "enhance_default": settings.enhance_default,
    }


@app.get("/api/health")
def health():
    return {"ok": True}


@app.post("/api/models/{key}/load")
async def load_model(key: str):
    if key not in PRESETS:
        return JSONResponse({"error": "unknown model"}, status_code=404)
    if not local_engine.available:
        return JSONResponse({"error": "local engine unavailable"}, status_code=400)
    threading.Thread(target=lambda: _safe_load(key), daemon=True).start()
    return {"ok": True}


def _safe_load(key: str) -> None:
    try:
        local_engine.ensure_loaded(key)
    except Exception as e:
        print(f"[local] load failed: {e}", flush=True)


def _line(obj: dict) -> bytes:
    return (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")


@app.post("/api/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    engine: str = Form("local"),
    model: str = Form(""),
    language: str = Form(""),
    prompt: str = Form(""),
    enhance: str = Form("true"),
    groq_key: str = Form(""),
):
    # Save the upload to disk so ffmpeg can seek in it (needed for many video containers).
    suffix = Path(file.filename or "upload").suffix[:10]
    fd, tmp_path = tempfile.mkstemp(prefix="ft_", suffix=suffix)
    with os.fdopen(fd, "wb") as out:
        await asyncio.to_thread(shutil.copyfileobj, file.file, out, 4 * 1024 * 1024)
    await file.close()

    language = language.strip() or None
    prompt = prompt.strip() or None
    model = model.strip() or settings.local_model
    do_enhance = enhance.lower() in {"1", "true", "on", "yes"}
    key = groq_key.strip() or settings.groq_api_key

    async def events():
        cancel = threading.Event()
        t0 = time.perf_counter()
        try:
            yield _line({"type": "status", "message": "Extracting audio…"})
            audio = await asyncio.to_thread(load_audio, tmp_path, do_enhance, settings.enhance_filter)
            duration = audio.size / SAMPLE_RATE
            yield _line({"type": "meta", "duration": round(duration, 2), "enhanced": do_enhance,
                         "extract_seconds": round(time.perf_counter() - t0, 2)})
            yield _line({"type": "status", "message": "Transcribing…"})

            if engine == "groq":
                gen = engine_groq.stream(audio, key, language, prompt)
            else:
                if not local_engine.available:
                    raise RuntimeError("Local engine is unavailable: " + (local_engine.describe()["import_error"] or "disabled in .env"))
                gen = local_engine.stream(audio, model, language, prompt, cancel)

            failed = False
            async for ev in gen:
                if ev.get("type") == "error":
                    failed = True
                yield _line(ev)
            if not failed:
                elapsed = time.perf_counter() - t0
                yield _line({"type": "done", "elapsed": round(elapsed, 2), "duration": round(duration, 2),
                             "speed": round(duration / elapsed, 1) if elapsed > 0 else None})
        except Exception as e:
            yield _line({"type": "error", "message": str(e) or type(e).__name__})
        finally:
            cancel.set()  # stops the local worker if the browser disconnected / pressed Stop
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    return StreamingResponse(
        events(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
