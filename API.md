# FastTranscribe API

Setup and run: see [README](README.md). Base URL: `http://<host>:8000` (interactive Swagger UI at `/docs`). Send any audio **or video** file; the
server extracts the audio with ffmpeg and streams the transcript back **segment by segment while it is still
being decoded**, so the UI can show text live.

## Server setup for an external frontend (`.env`)

| Variable | Purpose |
|---|---|
| `HOST=0.0.0.0` | Accept calls from other machines (default `127.0.0.1` = this PC only). |
| `CORS_ORIGINS=http://localhost:5173,https://app.example.com` | Browser origins allowed to call the API. Default `*` (any). Set it in production. |
| `API_KEY=long-secret` | Optional. If set, every `/api/*` call must send `X-API-Key: <key>` (or `Authorization: Bearer <key>`), otherwise `401 {"error": "invalid or missing API key"}`. The bundled `static/index.html` UI does not send a key, so leave it empty if you use that UI. |

## `POST /api/transcribe`: upload and stream

`multipart/form-data` fields:

| Field | Required | Default | Notes |
|---|---|---|---|
| `file` | yes | | Any audio/video format ffmpeg reads (mp3, wav, m4a, ogg, webm, mp4, mkv, mov…). |
| `engine` | no | `local` | `local` (your GPU/CPU) or `groq` (online). |
| `model` | no | server default | Local only: `large-v3-turbo`, `large-v3`, `hinglish`, `small` (see `GET /api/status` → `local.presets`). |
| `language` | no | auto-detect | ISO code, e.g. `en`, `hi`. |
| `prompt` | no | | Vocabulary hint (names, jargon) to improve accuracy. |
| `enhance` | no | `true` | Noise clean-up before transcription. |
| `groq_key` | no | server `.env` key | Groq only; per-request key override. |

Response: `200`, `Content-Type: application/x-ndjson`: **one JSON object per line**, flushed as soon as it is
ready. Every object has a `type`:

| `type` | Fields | Meaning |
|---|---|---|
| `status` | `message` | Human-readable progress ("Extracting audio…", "Transcribing…", model loading). |
| `meta` | `duration` (s), `enhanced`, `extract_seconds` | Audio extracted; total length known. Use `duration` for a progress bar. |
| `engine` | `engine`, `model`, `device`, `compute_type`, `note` | Which engine actually runs (device may fall back to `cpu`). |
| `info` | `language`, `language_probability` | Detected language (local engine). |
| `segment` | `start`, `end` (s), `text` | **The transcript.** Append these in order. |
| `progress` | `position` (s) | Groq engine only: audio position finished so far. |
| `done` | `elapsed`, `duration`, `speed` | Last line on success. `speed` = audio seconds per wall second. |
| `error` | `message` | Failure. Can arrive mid-stream after some segments; treat as terminal. |

Example stream:

```
{"type":"status","message":"Extracting audio…"}
{"type":"meta","duration":87.77,"enhanced":true,"extract_seconds":0.23}
{"type":"engine","engine":"local","model":"small","device":"cuda","compute_type":"int8_float16","note":null}
{"type":"info","language":"en","language_probability":0.955}
{"type":"segment","start":0.0,"end":4.81,"text":"Good morning, everyone."}
{"type":"segment","start":5.14,"end":32.21,"text":"We have to make sure that…"}
{"type":"done","elapsed":6.4,"duration":87.77,"speed":13.7}
```

**Cancelling:** abort the request (`AbortController.abort()` / close the connection). The server stops the job.
**Notes:** the upload is saved fully before processing starts, so `status` lines begin after the upload finishes.
Only one local job decodes at a time (others queue). Treat a stream that ends without `done` or `error` as
a dropped connection.

### JavaScript example (browser / React / Vue: no library needed)

`EventSource` can't POST files, so use `fetch` and read the body stream:

```js
async function transcribe(file, { onSegment, onEvent, signal, apiKey } = {}) {
  const form = new FormData();
  form.append("file", file);
  form.append("engine", "local");      // or "groq"
  // form.append("language", "hi");

  const res = await fetch("http://localhost:8000/api/transcribe", {
    method: "POST", body: form, signal,
    headers: apiKey ? { "X-API-Key": apiKey } : {},
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`);

  const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += value;
    const lines = buf.split("\n");
    buf = lines.pop();                       // keep the incomplete last line
    for (const line of lines) {
      if (!line.trim()) continue;
      const ev = JSON.parse(line);
      onEvent?.(ev);
      if (ev.type === "segment") onSegment?.(ev);
      if (ev.type === "error") throw new Error(ev.message);
    }
  }
}
```

### curl / Python

```bash
curl -N -H "X-API-Key: secret" -F "file=@meeting.mp4" -F engine=local http://localhost:8000/api/transcribe
```

```python
import httpx, json
with httpx.stream("POST", "http://localhost:8000/api/transcribe", files={"file": open("meeting.mp4", "rb")},
                  data={"engine": "local"}, timeout=None) as r:
    for line in r.iter_lines():
        if line: print(json.loads(line))
```

## Other endpoints

| Endpoint | Returns |
|---|---|
| `GET /api/health` | `{"ok": true}`: liveness check. |
| `GET /api/status` | `{local: {available, status, model, device, compute_type, cuda_devices, default_model, presets:[{key,label,note,vram_gb,downloaded}], error}, groq: {configured, model}, enhance_default}`. Use it to build the engine/model pickers and to see whether the model is still loading (`local.status`). |
| `POST /api/models/{key}/load` | `{"ok": true}`: starts loading/downloading a model in the background (`404` unknown key, `400` local engine unavailable). Poll `/api/status`. |

Errors before the stream starts use normal HTTP codes (`401` bad key, `422` missing `file`). Anything after
that arrives in-stream as an `error` event, because the HTTP status is already `200`.
