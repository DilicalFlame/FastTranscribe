"""Audio extraction (works for any audio or video file) and silence-aware chunking."""
from __future__ import annotations

import io
import shutil
import subprocess
import wave

import numpy as np

SAMPLE_RATE = 16_000


def ffmpeg_exe() -> str:
    """System ffmpeg if installed, otherwise the binary bundled by imageio-ffmpeg."""
    found = shutil.which("ffmpeg")
    if found:
        return found
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def load_audio(path: str, enhance: bool, enhance_filter: str) -> np.ndarray:
    """Decode the file's audio track to 16 kHz mono float32. Video streams are ignored."""
    cmd = [ffmpeg_exe(), "-nostdin", "-hide_banner", "-loglevel", "error",
           "-i", path, "-vn", "-sn", "-dn", "-map", "0:a:0?"]
    if enhance and enhance_filter:
        cmd += ["-af", enhance_filter]
    cmd += ["-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "s16le", "-acodec", "pcm_s16le", "-"]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        msg = proc.stderr.decode(errors="ignore").strip().splitlines()
        raise RuntimeError("ffmpeg could not read this file: " + (msg[-1] if msg else "unknown error"))
    pcm = np.frombuffer(proc.stdout, dtype=np.int16)
    if pcm.size == 0:
        raise RuntimeError("No audio track found in this file.")
    return pcm.astype(np.float32) / 32768.0


def split_on_silence(audio: np.ndarray, target_s: int, search_s: float = 8.0) -> list[tuple[int, int]]:
    """Cut into ~target_s pieces, moving each cut to the quietest 100 ms nearby so words aren't split."""
    n = audio.size
    step = target_s * SAMPLE_RATE
    if n <= step * 1.25:
        return [(0, n)]
    frame = SAMPLE_RATE // 10
    bounds = [0]
    pos = step
    while pos < n - step * 0.25:
        lo = max(bounds[-1] + frame, int(pos - search_s * SAMPLE_RATE))
        hi = min(n - frame, int(pos + search_s * SAMPLE_RATE))
        window = audio[lo:hi]
        k = window.size // frame
        if k > 0:
            rms = np.sqrt((window[: k * frame].reshape(k, frame) ** 2).mean(axis=1))
            cut = lo + int(rms.argmin()) * frame + frame // 2
        else:
            cut = pos
        bounds.append(cut)
        pos = cut + step
    bounds.append(n)
    return list(zip(bounds[:-1], bounds[1:]))


def is_silent(chunk: np.ndarray, threshold: float = 0.004) -> bool:
    return chunk.size == 0 or float(np.sqrt((chunk ** 2).mean())) < threshold


def to_wav_bytes(chunk: np.ndarray) -> bytes:
    pcm = (np.clip(chunk, -1.0, 1.0) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()
