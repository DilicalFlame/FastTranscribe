# FastTranscribe guide: models, accuracy, internals

## Models (local)

| Key | What | VRAM (int8) | When to use |
|---|---|---|---|
| `large-v3-turbo` | Whisper large-v3-turbo | ~1.6 GB | **Default.** Fast, 99 languages, good accuracy. |
| `large-v3` | Whisper large-v3 (full) | ~3 GB | Best accuracy on Hindi, accents and noisy rooms. About 3× slower than turbo. Fits a 4 GB card. |
| `hinglish` | Oriserve *Whisper-Hindi2Hinglish-Prime* (CT2 conversion) | ~1.6 GB | Hindi/Hinglish speech written in **Roman script** ("haan toh meeting start karte hain"). Experimental community conversion. |
| `small` | Whisper small | ~0.6 GB | Only for machines without a GPU. Much lower accuracy. |

Switch models from the dropdown. Only one model is kept in memory at a time.

### Expected speed
Rough estimates. Your numbers depend on GPU, audio and settings, and the app shows the real **× realtime** live.

| Hardware | large-v3-turbo | large-v3 |
|---|---|---|
| RTX 4060 (8 GB) | ~40–80× realtime (30 min → under a minute) | ~15–25× |
| 4 GB GPU (GTX 1650 / RTX 3050 laptop) | ~15–30× (30 min → 1–2 min) | ~6–12× |
| CPU only | ~1–3× | slower than realtime |
| Groq (online) | — | very fast; mostly limited by your upload speed |

Reference point: a published faster-whisper benchmark transcribed 13 min of audio with large-v3-turbo int8
in ~20 s on an RTX 2080 Ti, using ~1.5 GB of VRAM, *without* batching. This app uses batched decoding, which is faster.

---

## Getting the best accuracy on meetings and Hinglish

- **Enhance audio** (on by default) runs ffmpeg filters before transcription: removes rumble and hiss, light
  denoise, and evens out quiet and loud speakers (`dynaudnorm`). That helps with laptop-mic and phone recordings.
  For already-clean studio audio you can switch it off.
- **Vocabulary / context prompt**: type names, product terms and acronyms ("Rahul, Priya, Kubernetes, Q3 OKRs").
  A sample sentence in the style you want also steers the output. A Roman-script Hinglish line such as
  *"Haan toh, let's start the standup. Aaj ka agenda kya hai?"* nudges Whisper toward Roman Hinglish.
- **Script for Hindi**:
  - Whisper with Language = *Hindi / Hinglish* usually writes Hindi words in **Devanagari** and English words in English.
  - For consistent **Roman-script Hinglish**, use the `hinglish` local model. It forces the right setting by itself.
  - *Auto-detect* works for mixed meetings, but it guesses the language from the opening seconds.
    If a meeting is mostly Hindi, choose Hindi.
- **Model**: if turbo misses words, use `large-v3` locally, or Groq (which uses full `whisper-large-v3` by default).
- **Beam size** 5 (default) is more accurate on noisy audio. Set `LOCAL_BEAM_SIZE=1` in `.env` for maximum speed.

---

## How it works

```
browser ──upload──▶ FastAPI ──ffmpeg──▶ 16 kHz mono audio (+ optional clean-up filters)
   ▲                                   │
   │  NDJSON stream, one line per      ├─ Local: faster-whisper (Silero VAD + batched decoding).
   │  segment, sent the moment it's    │         Segments are pushed as soon as they're decoded.
   └──────── decoded ◀─────────────────┴─ Groq: cut at quiet points into ~2-min pieces, 3 sent in
                                                 parallel, results streamed back in order.
```

- API reference for integrators: see [API.md](API.md).
- Pressing **Stop** cancels the job on the server too.

```
run.py                 start server + open browser
server/main.py         API + streaming
server/engine_local.py faster-whisper engine (GPU detection, fallbacks, batching)
server/engine_groq.py  Groq engine (chunking, parallel requests, retries)
server/audio.py        ffmpeg extraction, silence-aware chunking
server/models.py       model presets + downloads
static/index.html      the whole frontend
scripts/               download_model.py, check_gpu.py
```

## Limitations / ideas
- No speaker labels (who said what). That would need a diarization model such as pyannote; it's a natural next step.
- Groq mode sends your audio to Groq's servers. Use Local mode for confidential meetings.
- The `hinglish` model is a third-party conversion; validate it on your own recordings.
