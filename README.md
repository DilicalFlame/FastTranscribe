# FastTranscribe

Drop an audio or video file in the browser and the transcript appears line by line **while it is
being processed**. You don't wait for a progress bar to finish. Built for meeting recordings,
including noisy audio and Hindi/Hinglish.

- **Local mode**: [faster-whisper](https://github.com/SYSTRAN/faster-whisper) on your NVIDIA GPU (or CPU). Private, free, no limits.
- **Online mode**: [Groq](https://console.groq.com)'s free Whisper API. No GPU needed. Audio is sent to Groq.
- Video files: only the audio track is used (ffmpeg is bundled, nothing to install).
- **Hide timestamps** toggle. Export **.srt** with timestamps, or **.txt** without them (the export button follows the toggle).
- Live stats: position, elapsed time, **× realtime** speed, time left.

**Docs:** [API.md](API.md) (integrating a frontend) · [GUIDE.md](GUIDE.md) (models, accuracy tips, how it works)

---

## 1. Setup (one time)

You need **[uv](https://docs.astral.sh/uv/getting-started/installation/)**. It installs the right Python by itself.

```powershell
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
```bash
# Linux / macOS
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then, inside this folder:

```bash
uv sync --extra cuda      # with an NVIDIA GPU (adds cuBLAS + cuDNN 9 as pip packages)
# or
uv sync                   # CPU-only machine, Mac, or if you'll only use Groq
```

Copy the settings file:

```bash
cp .env.example .env      # Windows: copy .env.example .env
```

### Online mode (Groq): add your key
1. Sign up free at **https://console.groq.com** → **API Keys** → *Create API Key* (no card needed).
2. Put it in `.env`:  `GROQ_API_KEY=gsk_...`
   or paste it in the app under **Groq API key** (it is kept in your browser only).

Free tier at the time of writing: ~2 hours of audio per hour, ~8 hours per day, 20 requests/min.
The app cuts audio into ~2-minute pieces and retries automatically when rate-limited.

### Local mode: get a model
Models download automatically on first use. To download ahead of time (recommended, the turbo model is ~1.6 GB):

```bash
uv run scripts/download_model.py               # the default (large-v3-turbo)
uv run scripts/download_model.py large-v3 hinglish
uv run scripts/download_model.py --list        # see all + what's downloaded
```

Models are saved in `./models/`. Check that your GPU works:

```bash
uv run scripts/check_gpu.py
```

## 2. Run

```bash
uv run run.py
```

It opens **http://127.0.0.1:8000**. Pick **Local** or **Online (Groq)**, drop a file, and watch.
(Windows: you can also double-click `start.bat`.)

---

## GPU setup / troubleshooting

The local engine needs an **NVIDIA GPU + recent driver** and **cuBLAS + cuDNN 9 (CUDA 12)**.
`uv sync --extra cuda` installs those as pip packages, and the app loads them automatically.
You do **not** need to install the CUDA Toolkit.

### Windows: `Library cublas64_12.dll is not found or cannot be loaded`

Cause: the NVIDIA runtime libs weren't installed in `.venv` (a plain `uv sync` skips the `cuda` extra).
Fix, on a fresh laptop (after installing the NVIDIA driver and `uv`):

```powershell
uv sync --extra cuda                # installs cuBLAS 12 + cuDNN 9 into .venv\Lib\site-packages\nvidia
uv run scripts/check_gpu.py         # should print "small ready on cuda (int8_float16)"
```

`pyproject.toml` pins `nvidia-cublas-cu12<12.9.2` because 12.9.3.x ships Linux-only wheels, which makes
`uv sync --extra cuda` fail on Windows with "doesn't have a source distribution or wheel for the current
platform". Don't remove the pin until NVIDIA publishes Windows wheels for newer versions.

Always start the server through `uv run` (or `.venv\Scripts\python`) so it uses this environment.
If `check_gpu.py` ends with a `charmap codec can't encode '→'` error, that is only console encoding:
run `$env:PYTHONUTF8=1` first. The GPU itself is fine.

- **"GPU failed … running on CPU"** in the app → run `uv sync --extra cuda`, update the NVIDIA driver,
  then `uv run scripts/check_gpu.py`.
- **Out of memory on a 4 GB card** → the app automatically continues in sequential mode. To avoid it,
  set `LOCAL_BATCH_SIZE=4` (or use `large-v3-turbo` instead of `large-v3`).
- **GTX 10-series (Pascal)** → no fast fp16, so the app picks `int8` automatically.
- **Mac / AMD GPU** → local mode runs on CPU (slow). Use Groq for speed.
- **Model download blocked or slow** → download the model folder from Hugging Face manually into
  `models/<key>/` (e.g. `models/large-v3-turbo/model.bin` …), or set `LOCAL_MODEL=/path/to/folder`.
  Repos: `mobiuslabsgmbh/faster-whisper-large-v3-turbo`, `Systran/faster-whisper-large-v3`,
  `Hub84/faster-whisper-hinglish-prime`, `Systran/faster-whisper-small`.

All settings are in `.env` (see `.env.example` for every option).
