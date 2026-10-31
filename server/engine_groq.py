"""Online transcription with Groq's free Whisper API.

Groq returns a whole transcript per request, so the audio is cut into ~2-minute pieces at
quiet points, a few pieces are sent in parallel, and results are streamed back in order.
"""
from __future__ import annotations

import asyncio
from typing import AsyncIterator

import httpx
import numpy as np

from .audio import SAMPLE_RATE, is_silent, split_on_silence, to_wav_bytes
from .config import settings


class GroqError(RuntimeError):
    pass


async def _transcribe_chunk(client: httpx.AsyncClient, key: str, wav: bytes, offset: float,
                            language: str | None, prompt: str | None, model: str) -> list[dict]:
    data = {"model": model, "response_format": "verbose_json", "temperature": "0"}
    if language:
        data["language"] = language
    if prompt:
        data["prompt"] = prompt[:800]  # Groq caps the prompt at 224 tokens
    for attempt in range(8):
        try:
            r = await client.post(
                f"{settings.groq_base_url}/audio/transcriptions",
                headers={"Authorization": f"Bearer {key}"},
                files={"file": ("chunk.wav", wav, "audio/wav")},
                data=data,
            )
        except httpx.TransportError as e:
            if attempt >= 3:
                raise GroqError(f"Network error talking to Groq: {e}") from e
            await asyncio.sleep(2 * (attempt + 1))
            continue
        if r.status_code == 200:
            out = []
            for s in r.json().get("segments") or []:
                text = (s.get("text") or "").strip()
                # drop Whisper's classic hallucinations on silence/noise
                if not text or (s.get("no_speech_prob", 0) > 0.8 and s.get("avg_logprob", 0) < -0.8):
                    continue
                out.append({"type": "segment", "start": round(offset + float(s["start"]), 2),
                            "end": round(offset + float(s["end"]), 2), "text": text})
            return out
        if r.status_code == 429 or r.status_code >= 500:
            wait = r.headers.get("retry-after")
            try:
                delay = float(wait) if wait else 5.0 * (attempt + 1)
            except ValueError:
                delay = 10.0
            await asyncio.sleep(min(delay, 90))
            continue
        try:
            detail = r.json().get("error", {}).get("message", r.text)
        except Exception:
            detail = r.text
        if r.status_code == 401:
            raise GroqError("Groq rejected the API key (401). Check GROQ_API_KEY in .env or the key in Settings.")
        raise GroqError(f"Groq error {r.status_code}: {detail}")
    raise GroqError("Groq kept rate-limiting this request. Free tier: ~2 h of audio per hour, 8 h per day.")


async def stream(audio: np.ndarray, key: str, language: str | None,
                 prompt: str | None) -> AsyncIterator[dict]:
    if not key:
        raise GroqError("No Groq API key. Put GROQ_API_KEY in .env, or paste a key in Settings.")
    model = settings.groq_model
    yield {"type": "engine", "engine": "groq", "model": model}

    pieces = split_on_silence(audio, settings.groq_chunk_seconds)
    sem = asyncio.Semaphore(max(1, settings.groq_parallel))
    timeout = httpx.Timeout(180.0, connect=15.0)

    async with httpx.AsyncClient(timeout=timeout) as client:
        async def job(a: int, b: int) -> list[dict]:
            chunk = audio[a:b]
            if is_silent(chunk):
                return []
            async with sem:
                return await _transcribe_chunk(client, key, to_wav_bytes(chunk), a / SAMPLE_RATE,
                                               language, prompt, model)

        tasks = [asyncio.create_task(job(a, b)) for a, b in pieces]
        try:
            for i, t in enumerate(tasks):
                for seg in await t:      # await in order → transcript streams in order
                    yield seg
                yield {"type": "progress", "position": round(pieces[i][1] / SAMPLE_RATE, 2)}
        finally:
            for t in tasks:
                t.cancel()
