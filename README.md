<div align="center">

# Whisper Subtitle Generator

**Video and audio in, clean SRT out. Self-hosted, on one GPU.**

![Python](https://img.shields.io/badge/Python-3.11+-3776ab?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![Whisper](https://img.shields.io/badge/Whisper-large--v3-412991?logo=openai&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-ee4c2c?logo=pytorch&logoColor=white)
[![tests](https://github.com/AkoSuminoe/whisper-subtitle-generator/actions/workflows/tests.yml/badge.svg)](https://github.com/AkoSuminoe/whisper-subtitle-generator/actions/workflows/tests.yml)
![Coverage](https://img.shields.io/badge/coverage-90%25-2fa572)
![License](https://img.shields.io/badge/License-MIT-a3e635)

**[Live demo](https://whisper.berktan.dev)**

</div>

---

## About

Whisper transcribes well and formats badly. Raw output arrives as cues broken in the wrong places, at a pace nobody can read, with every proper noun spelled the way it sounded.

This project fixes the formatting and then does the harder job: exposing the result to the public from a single consumer GPU on a domestic connection, without letting a stranger occupy the machine.

Three pieces:

| Piece | What it is |
| --- | --- |
| `app.py`, `subtitle_core.py` | The desktop application and the transcription and clean-up logic |
| `server/main.py` | A FastAPI service wrapping the same core with a queue, limits and a captcha |
| `web/SubtitleDemo.tsx` | A reference React component for the HTTP API |

---

## Features

### Subtitle clean-up

- Cues are merged and split so they read at a sensible pace instead of matching the model's segment boundaries.
- A terms file corrects the proper nouns the model reliably mishears. Copy `terms.example.json` to `terms.json` to start your own.
- Turkish and English.

### Built to be exposed

The limits are the interesting part, because a public demo on one GPU is a machine anyone can occupy. They are layered rather than singular:

- **Per request**: maximum upload size and maximum audio duration.
- **Per client**: hourly and daily rates, keyed on `cf-connecting-ip`.
- **Global**: ceilings on queue depth, jobs per hour and day, transcribed minutes per day, and disk. These persist to `budget.json`, because a limit a process forgets when it crashes is not a limit.
- **Captcha**: Cloudflare Turnstile, verified server side before a job is accepted.
- Jobs have a timeout and a TTL. Uploads and generated subtitles are deleted when a job expires.

Checks run cheapest first, so abuse is rejected before it costs anything: the API key, then the declared content length, then queue depth and disk, then the client rate limit, and only then the captcha, which is the one step that makes an outbound call.

---

## API

| Endpoint | Purpose |
| --- | --- |
| `GET /api/health` | Device, model, queue depth, whether a captcha is required, the active limits and the current budget |
| `POST /api/transcribe` | Multipart upload. Returns a `job_id` |
| `GET /api/jobs/{id}` | Status, stage, progress, queue position and ETA |
| `GET /api/jobs/{id}/srt` | The finished subtitles. `?download=1` for an attachment |
| `DELETE /api/jobs/{id}` | Discard a job and its files |

`/api/health` is worth calling from a front end rather than hard-coding anything: it reports the server's real limits, so the interface can describe them accurately instead of drifting from the configuration.

---

## Getting Started

**1. Install PyTorch first**, from the CUDA index, or pip resolves a CPU-only build and transcription becomes unusably slow:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

**2. Install the rest**

```bash
pip install -r server/requirements.txt
```

**3. Configure**

```bash
cp server/.env.example server/.env
```

**4. Run**

```bash
uvicorn server.main:app --host 127.0.0.1 --port 8000
```

The desktop application is separate and needs no server: run `start.bat`, or `python app.py`.

`ffmpeg` must be on `PATH`.

---

## Tests

```bash
pip install -r requirements-dev.txt
pytest --cov
```

The suite needs **no GPU and no network**. Torch and Whisper are replaced with stubs, so what runs is the part worth testing: the clean-up pipeline, the limits, and the queue. It finishes in about five seconds. One test shells out to `ffprobe` for real and is skipped when ffmpeg is absent.

| Module | Coverage |
| --- | --- |
| `subtitle_core.py` | 94% |
| `server/main.py` | 88% |
| **Total** | **90%** — 88 tests |

Every push runs the suite on Python 3.11, 3.12 and 3.13 via GitHub Actions; the badge above is that workflow, not a number typed into this file. The run fails if coverage drops below 88%, so the figure in the table cannot quietly rot: if it falls, the badge goes red. Each run also publishes a coverage table to its job summary.

| File | Covers |
| --- | --- |
| `tests/test_pipeline.py` | Turkish casing, the terms dictionary, cue layout, the drop rules, SRT rendering |
| `tests/test_server.py` | Upload validation, every limit, the captcha, queue reporting, worker failure |
| `tests/test_progress_hook.py` | The tqdm swap that produces real progress, and its restoration |
| `tests/test_backend.py` | Device detection, the single-slot model cache, budget bookkeeping |

Two properties are worth singling out, because they are the ones a reader is entitled to doubt:

- `test_global_cap_holds_under_ip_rotation` accepts up to the global ceiling, then makes four further requests from four different addresses and asserts every one is refused. Per-client limits are defeated by rotation; this is the limit that is not.
- `test_budget_survives_a_restart` fills the hourly cap, reloads the module from the persisted file, and asserts the cap is still full. A limit a process forgets when it crashes is not a limit.

`test_probe_duration_reads_a_real_file` earns its keep for a different reason: it runs `ffprobe` rather than stubbing it, and exists because a Windows-only `creationflags` argument was being passed unconditionally. On Linux that raised, the exception handler swallowed it, and every duration came back `None` — which would have rejected every upload with "could not read this file". A stubbed test cannot see that; CI on Ubuntu can.

### What is not covered

The 10% is almost entirely code that only runs with real hardware or on a real clock: loading Whisper onto the GPU, the transcription loop itself, the hourly cleanup thread, and the job watchdog's timeout branch. `app.py` is excluded altogether — it is a Tkinter GUI, and testing it would mean testing Tk.

---

## Deployment

The live demo runs on a Windows machine behind a Cloudflare Tunnel, with the API and the web front end sharing a single hostname:

```yaml
ingress:
  - hostname: whisper.berktan.dev
    path: ^/api/.*
    service: http://localhost:8000
  - hostname: whisper.berktan.dev
    service: http://localhost:3000
```

Path rules are evaluated in order, so the API rule has to come first.

Sharing a hostname this way is what removes CORS from the design: the browser only ever talks to one origin and the split happens inside the tunnel, so there is no preflight and no origin allowlist to keep in step with the front end. `ALLOWED_ORIGINS` is still set, as defence in depth rather than as the control.

Two settings matter more than they look:

- `HOST=127.0.0.1`, not `0.0.0.0`. The tunnel connects from the same machine, so binding every interface only exposes the server to the local network. It is also what makes trusting `cf-connecting-ip` sound: if the port were reachable directly, anyone could set that header and rotate through fake values to defeat the per-client limits.
- `ALLOW_MODEL_OVERRIDE=false` in production, so a caller cannot ask for a larger model than you intended to pay for.

---

## Configuration

Everything is environment driven; see `server/.env.example` for the full list with defaults. The ones worth deciding deliberately:

| Variable | Why it matters |
| --- | --- |
| `TURNSTILE_SECRET` | Empty disables the captcha entirely |
| `MAX_UPLOAD_MB`, `MAX_DURATION_SEC` | The per-request ceiling |
| `RATE_LIMIT_PER_HOUR`, `RATE_LIMIT_PER_DAY` | Per client, and best effort: IP rotation defeats them |
| `GLOBAL_*` | The real ceiling, and the only limits an attacker cannot rotate around |
| `MAX_DISK_MB` | Stops a queue of large uploads filling the drive |
| `ALLOW_MODEL_OVERRIDE` | Leave false unless you want callers choosing the model |

---

## License

Released under the [MIT License](LICENSE).

Whisper models are downloaded at runtime and carry their own terms.
