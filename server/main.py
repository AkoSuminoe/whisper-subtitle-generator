"""FastAPI service exposing the subtitle pipeline.

One GPU means one job at a time, so work goes through a single worker thread.
Every limit below is enforced server-side; the global budget is persisted so a
restart cannot be used to reset it.
"""

import json
import os
import queue
import re
import sys
import tempfile
import threading
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import httpx
from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

SERVER_DIR = Path(__file__).resolve().parent

# Load server/.env before anything reads os.getenv below. Without this the file
# is silently ignored and the captcha stays off while looking configured.
try:
    from dotenv import load_dotenv

    load_dotenv(SERVER_DIR / ".env")
except ImportError:
    pass

# subtitle_core lives one level up, beside app.py.
sys.path.insert(0, str(SERVER_DIR.parent))

from subtitle_core import (  # noqa: E402
    LANGUAGES,
    MODEL_CHOICES,
    TermReplacer,
    build_srt,
    configure_utf8_console,
    load_terms,
    postprocess_segments,
    probe_duration,
    validate_terms,
    whisper_cache_dir,
    whisper_progress,
)

configure_utf8_console()

# --- Configuration (environment variables) -----------------------------------------
DEFAULT_MODEL = os.getenv("WHISPER_MODEL", "large-v3")
ALLOW_MODEL_OVERRIDE = os.getenv("ALLOW_MODEL_OVERRIDE", "false").lower() == "true"
TERMS_FILE = os.getenv("TERMS_FILE", "").strip()
API_KEY = os.getenv("API_KEY", "").strip()
ALLOWED_ORIGINS = [
    o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()
]

# Per-request limits.
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "100"))
MAX_DURATION_SEC = int(os.getenv("MAX_DURATION_SEC", "600"))

# Per-client limits (best effort: a client that rotates IPs defeats these,
# which is why the global budget below exists).
RATE_LIMIT_PER_HOUR = int(os.getenv("RATE_LIMIT_PER_HOUR", "3"))
RATE_LIMIT_PER_DAY = int(os.getenv("RATE_LIMIT_PER_DAY", "10"))

# Global limits. These are the hard ceiling on what the machine can be made to
# do, no matter how many clients or IP addresses ask.
MAX_QUEUE_DEPTH = int(os.getenv("MAX_QUEUE_DEPTH", "5"))
GLOBAL_JOBS_PER_HOUR = int(os.getenv("GLOBAL_JOBS_PER_HOUR", "20"))
GLOBAL_JOBS_PER_DAY = int(os.getenv("GLOBAL_JOBS_PER_DAY", "100"))
GLOBAL_AUDIO_MINUTES_PER_DAY = int(os.getenv("GLOBAL_AUDIO_MINUTES_PER_DAY", "180"))
MAX_DISK_MB = int(os.getenv("MAX_DISK_MB", "1000"))
JOB_TIMEOUT_SEC = int(os.getenv("JOB_TIMEOUT_SEC", "1800"))
JOB_TTL_SEC = int(os.getenv("JOB_TTL_SEC", "3600"))

# A caller may send their own dictionary. It is held in memory for the job and
# never written to disk, so nothing survives the request. The caps bound both
# memory and the size of the alternation regex built from it.
MAX_TERMS_BYTES = int(os.getenv("MAX_TERMS_BYTES", "8192"))
MAX_TERMS_ENTRIES = int(os.getenv("MAX_TERMS_ENTRIES", "200"))
MAX_TERM_LENGTH = int(os.getenv("MAX_TERM_LENGTH", "100"))

# Cloudflare Turnstile. Leave the secret empty to disable the check.
TURNSTILE_SECRET = os.getenv("TURNSTILE_SECRET", "").strip()
TURNSTILE_VERIFY_URL = "https://challenges.cloudflare.com/turnstile/v0/siteverify"

BUDGET_FILE = Path(os.getenv("BUDGET_FILE") or SERVER_DIR / "budget.json")
UPLOAD_DIR = Path(tempfile.gettempdir()) / "whisper_api_uploads"

MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
MAX_DISK_BYTES = MAX_DISK_MB * 1024 * 1024
ALLOWED_EXTENSIONS = {
    ".mp4", ".mov", ".mkv", ".avi", ".webm",
    ".mp3", ".wav", ".m4a", ".ogg", ".flac",
}

app = FastAPI(title="Whisper Subtitle Generator API", version="1.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


# --- Job state ----------------------------------------------------------------------

@dataclass
class Job:
    id: str
    filename: str
    language: str
    model: str
    status: str = "queued"  # queued | running | done | error
    stage: str = "Queued"
    progress: float = 0.0
    duration: Optional[float] = None
    error: Optional[str] = None
    srt: Optional[str] = None
    cue_count: int = 0
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    source_path: Optional[str] = None
    terms: Optional[dict] = None


_jobs = {}
_jobs_lock = threading.Lock()
_work_queue = queue.Queue()
_queue_order = deque()  # job ids waiting, for position and ETA

_hits_hour = defaultdict(deque)
_hits_day = defaultdict(deque)
_hits_lock = threading.Lock()

_backend = {"torch": None, "whisper": None, "device": "cpu", "gpu_name": None}
_model_cache = {"key": None, "model": None}
_backend_lock = threading.Lock()

# Observed transcription speed (audio seconds per wall second), for queue ETAs.
_speed = {"factor": 6.0}


# --- Global budget (persisted so a restart cannot reset it) --------------------------

_budget_lock = threading.Lock()
_budget = {"jobs": [], "audio": []}  # [ts, ...] and [[ts, seconds], ...]


def _load_budget() -> None:
    global _budget
    try:
        data = json.loads(BUDGET_FILE.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            _budget = {
                "jobs": [float(t) for t in data.get("jobs", [])],
                "audio": [[float(t), float(s)] for t, s in data.get("audio", [])],
            }
    except (OSError, ValueError, TypeError):
        _budget = {"jobs": [], "audio": []}


def _save_budget() -> None:
    try:
        BUDGET_FILE.write_text(json.dumps(_budget), encoding="utf-8")
    except OSError:
        pass  # a read-only disk must not break transcription


def _prune_budget(now: float) -> None:
    _budget["jobs"] = [t for t in _budget["jobs"] if now - t <= 86400]
    _budget["audio"] = [entry for entry in _budget["audio"] if now - entry[0] <= 86400]


def _budget_snapshot() -> dict:
    now = time.time()
    with _budget_lock:
        _prune_budget(now)
        jobs_hour = sum(1 for t in _budget["jobs"] if now - t <= 3600)
        jobs_day = len(_budget["jobs"])
        audio_day = sum(entry[1] for entry in _budget["audio"])
    return {
        "jobs_this_hour": jobs_hour,
        "jobs_today": jobs_day,
        "audio_minutes_today": round(audio_day / 60, 1),
    }


def _reserve_budget(audio_seconds: float) -> None:
    """Claim one job plus its audio time, or reject. The hard global ceiling."""
    now = time.time()
    with _budget_lock:
        _prune_budget(now)
        jobs_hour = sum(1 for t in _budget["jobs"] if now - t <= 3600)
        if jobs_hour >= GLOBAL_JOBS_PER_HOUR:
            raise HTTPException(429, "The service is at its hourly capacity. Try again later.")
        if len(_budget["jobs"]) >= GLOBAL_JOBS_PER_DAY:
            raise HTTPException(429, "The service is at its daily capacity. Try again tomorrow.")

        audio_day = sum(entry[1] for entry in _budget["audio"])
        if audio_day + audio_seconds > GLOBAL_AUDIO_MINUTES_PER_DAY * 60:
            raise HTTPException(429, "The service is at its daily audio limit. Try again tomorrow.")

        _budget["jobs"].append(now)
        _budget["audio"].append([now, audio_seconds])
        _save_budget()


# --- Guards ---------------------------------------------------------------------------

def _client_key(request: Request) -> str:
    forwarded = request.headers.get("cf-connecting-ip") or request.headers.get(
        "x-forwarded-for", ""
    )
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _check_rate_limit(key: str) -> None:
    now = time.time()
    with _hits_lock:
        for window, store, cap in (
            (3600, _hits_hour, RATE_LIMIT_PER_HOUR),
            (86400, _hits_day, RATE_LIMIT_PER_DAY),
        ):
            if cap <= 0:
                continue
            hits = store[key]
            while hits and now - hits[0] > window:
                hits.popleft()
            if len(hits) >= cap:
                retry = int(window - (now - hits[0]))
                unit = "hour" if window == 3600 else "day"
                raise HTTPException(
                    429, f"Limit reached ({cap} per {unit}). Try again in {retry}s."
                )
        _hits_hour[key].append(now)
        _hits_day[key].append(now)


def _require_api_key(provided: Optional[str]) -> None:
    if API_KEY and (provided or "").strip() != API_KEY:
        raise HTTPException(401, "Invalid or missing API key.")


async def _verify_turnstile(token: str, ip: str) -> None:
    """Validate a Cloudflare Turnstile token. Tokens are single-use at Cloudflare."""
    if not TURNSTILE_SECRET:
        return
    if not token:
        raise HTTPException(403, "Captcha required.")
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(
                TURNSTILE_VERIFY_URL,
                data={"secret": TURNSTILE_SECRET, "response": token, "remoteip": ip},
            )
        data = response.json()
    except (httpx.HTTPError, ValueError):
        # Fail closed: an unverifiable request must not reach the GPU.
        raise HTTPException(503, "Captcha verification unavailable. Try again shortly.")
    if not data.get("success"):
        raise HTTPException(403, "Captcha verification failed. Reload and try again.")


def _parse_request_terms(raw: str):
    """Validate a caller-supplied terms dictionary, or return None if absent."""
    raw = (raw or "").strip()
    if not raw or raw == "{}":
        return None

    if len(raw.encode("utf-8")) > MAX_TERMS_BYTES:
        raise HTTPException(413, f"The terms dictionary exceeds {MAX_TERMS_BYTES} bytes.")

    try:
        parsed = json.loads(raw)
    except ValueError as exc:
        raise HTTPException(400, f"The terms dictionary is not valid JSON: {exc}")

    terms, _warning = validate_terms(parsed)
    if not terms:
        raise HTTPException(400, 'Terms must be a flat {"wrong": "correct"} object.')
    if len(terms) > MAX_TERMS_ENTRIES:
        raise HTTPException(413, f"At most {MAX_TERMS_ENTRIES} terms are allowed.")

    for key, value in terms.items():
        if len(key) > MAX_TERM_LENGTH or len(value) > MAX_TERM_LENGTH:
            raise HTTPException(
                413, f"Each term must be at most {MAX_TERM_LENGTH} characters."
            )
    return terms


def _check_disk_quota() -> None:
    try:
        used = sum(f.stat().st_size for f in UPLOAD_DIR.glob("*") if f.is_file())
    except OSError:
        return
    if used >= MAX_DISK_BYTES:
        raise HTTPException(503, "The service is busy. Try again shortly.")


def _check_queue_depth() -> None:
    with _jobs_lock:
        waiting = sum(1 for j in _jobs.values() if j.status in ("queued", "running"))
    if waiting >= MAX_QUEUE_DEPTH:
        raise HTTPException(503, f"The queue is full ({MAX_QUEUE_DEPTH}). Try again shortly.")


# --- Backend --------------------------------------------------------------------------

def _ensure_backend():
    """Import torch/whisper once and pick the device. Worker thread only."""
    with _backend_lock:
        if _backend["whisper"] is None:
            import torch
            import whisper

            _backend["torch"] = torch
            _backend["whisper"] = whisper
            if torch.cuda.is_available():
                _backend["device"] = "cuda"
                _backend["gpu_name"] = torch.cuda.get_device_name(0)
            else:
                _backend["device"] = "cpu"
    return _backend


def _load_model(name: str):
    """Single-slot model cache, mirroring the desktop app's VRAM behaviour."""
    device = _backend["device"]
    key = (name, device)
    if _model_cache["key"] == key and _model_cache["model"] is not None:
        return _model_cache["model"]

    _model_cache["model"] = None
    _model_cache["key"] = None
    if device == "cuda":
        _backend["torch"].cuda.empty_cache()

    model = _backend["whisper"].load_model(name, device=device)
    _model_cache["model"] = model
    _model_cache["key"] = key
    return model


def _model_is_cached(name: str) -> bool:
    try:
        url = _backend["whisper"]._MODELS[name]
        return os.path.exists(os.path.join(whisper_cache_dir(), os.path.basename(url)))
    except Exception:
        return True


def _update(job_id: str, **fields) -> None:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job:
            for key, value in fields.items():
                setattr(job, key, value)


class JobTimeout(RuntimeError):
    pass


def _worker_loop() -> None:
    """Process one job at a time: a single GPU cannot usefully do more."""
    while True:
        job_id = _work_queue.get()
        try:
            _queue_order.remove(job_id)
        except ValueError:
            pass
        try:
            _process_job(job_id)
        except Exception as exc:  # never let the worker thread die
            _update(
                job_id,
                status="error",
                stage="Failed",
                error=f"{type(exc).__name__}: {exc}",
            )
        finally:
            _work_queue.task_done()


def _process_job(job_id: str) -> None:
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None or job.source_path is None:
        return

    source = Path(job.source_path)
    started = time.time()
    try:
        _update(job_id, status="running", stage="Loading model", progress=0.0,
                started_at=started)
        _ensure_backend()

        if not _model_is_cached(job.model):
            _update(job_id, stage=f"Downloading {job.model} (first run only)")
        model = _load_model(job.model)

        _update(job_id, stage="Transcribing", progress=0.0)
        transcribe_started = time.time()

        def on_progress(fraction: float) -> None:
            # Whisper calls this as it seeks, which doubles as a watchdog tick.
            if time.time() - transcribe_started > JOB_TIMEOUT_SEC:
                raise JobTimeout("Job exceeded its time limit and was stopped.")
            _update(job_id, progress=round(float(fraction), 4))

        with whisper_progress(on_progress):
            result = model.transcribe(
                str(source),
                language=job.language,
                fp16=(_backend["device"] == "cuda"),
                verbose=False,
            )

        elapsed = max(0.001, time.time() - transcribe_started)
        if job.duration:
            observed = job.duration / elapsed
            _speed["factor"] = _speed["factor"] * 0.7 + observed * 0.3

        _update(job_id, stage="Cleaning transcript", progress=1.0)
        # A caller-supplied dictionary applies to this job only.
        if job.terms is not None:
            terms = job.terms
        else:
            terms, _warning = load_terms(Path(TERMS_FILE) if TERMS_FILE else None)
        cues = postprocess_segments(
            result.get("segments", []), TermReplacer(terms, job.language)
        )
        srt = build_srt(cues)

        _update(job_id, status="done", stage="Done", progress=1.0, srt=srt,
                cue_count=len(cues))
    finally:
        try:
            source.unlink(missing_ok=True)
        except OSError:
            pass
        # The upload and the caller's dictionary are both needed only while the
        # job runs; neither outlives it.
        _update(job_id, source_path=None, terms=None)


def _cleanup_loop() -> None:
    """Drop aged-out jobs and any orphaned uploads."""
    while True:
        time.sleep(60)
        cutoff = time.time() - JOB_TTL_SEC
        with _jobs_lock:
            stale = [key for key, job in _jobs.items() if job.created_at < cutoff]
            for job_id in stale:
                job = _jobs.pop(job_id)
                if job.source_path:
                    try:
                        Path(job.source_path).unlink(missing_ok=True)
                    except OSError:
                        pass
        try:
            for leftover in UPLOAD_DIR.glob("*"):
                if leftover.is_file() and leftover.stat().st_mtime < cutoff:
                    leftover.unlink(missing_ok=True)
        except OSError:
            pass


def _estimate_wait(job_id: str) -> float:
    """Seconds of audio ahead in the line, converted by observed speed."""
    with _jobs_lock:
        ahead = 0.0
        for other in _jobs.values():
            if other.id == job_id:
                continue
            if other.status == "running":
                remaining = (other.duration or 0) * (1 - other.progress)
                ahead += remaining
            elif other.status == "queued" and other.created_at < _jobs[job_id].created_at:
                ahead += other.duration or 0
    return round(ahead / max(0.5, _speed["factor"]), 1)


@app.on_event("startup")
def _startup() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    _load_budget()
    threading.Thread(target=_worker_loop, daemon=True).start()
    threading.Thread(target=_cleanup_loop, daemon=True).start()


# --- Routes ---------------------------------------------------------------------------

@app.get("/api/health")
def health():
    with _jobs_lock:
        active = sum(1 for j in _jobs.values() if j.status in ("queued", "running"))
    return {
        "status": "ok",
        "device": _backend["device"],
        "gpu": _backend["gpu_name"],
        "backend_loaded": _backend["whisper"] is not None,
        "default_model": DEFAULT_MODEL,
        "models": MODEL_CHOICES if ALLOW_MODEL_OVERRIDE else [DEFAULT_MODEL],
        "languages": LANGUAGES,
        "queue_depth": active,
        "captcha_required": bool(TURNSTILE_SECRET),
        "limits": {
            "max_upload_mb": MAX_UPLOAD_MB,
            "max_duration_sec": MAX_DURATION_SEC,
            "rate_limit_per_hour": RATE_LIMIT_PER_HOUR,
            "rate_limit_per_day": RATE_LIMIT_PER_DAY,
            "max_queue_depth": MAX_QUEUE_DEPTH,
            "max_terms_entries": MAX_TERMS_ENTRIES,
            "max_terms_bytes": MAX_TERMS_BYTES,
            "api_key_required": bool(API_KEY),
        },
        "budget": _budget_snapshot(),
    }


@app.post("/api/transcribe")
async def transcribe(
    request: Request,
    file: UploadFile = File(...),
    language: str = Form("tr"),
    model: str = Form(""),
    turnstile_token: str = Form(""),
    terms: str = Form(""),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
):
    # Cheapest checks first, so abuse is rejected before it costs anything.
    _require_api_key(x_api_key)

    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > MAX_UPLOAD_BYTES + 1024 * 1024:
        raise HTTPException(413, f"File exceeds the {MAX_UPLOAD_MB} MB limit.")

    _check_queue_depth()
    _check_disk_quota()

    client = _client_key(request)
    _check_rate_limit(client)
    await _verify_turnstile(turnstile_token, client)

    if language not in LANGUAGES.values():
        allowed = sorted(set(LANGUAGES.values()))
        raise HTTPException(400, f"Unsupported language. Use one of {allowed}.")

    chosen = (model or DEFAULT_MODEL).strip()
    if not ALLOW_MODEL_OVERRIDE:
        chosen = DEFAULT_MODEL
    elif chosen not in MODEL_CHOICES:
        raise HTTPException(400, f"Unknown model. Use one of {MODEL_CHOICES}.")

    request_terms = _parse_request_terms(terms)

    suffix = Path(file.filename or "audio").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise HTTPException(400, f"Unsupported file type. Allowed: {allowed}.")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    job_id = uuid.uuid4().hex
    target = UPLOAD_DIR / f"{job_id}{suffix}"

    written = 0
    try:
        with target.open("wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, f"File exceeds the {MAX_UPLOAD_MB} MB limit.")
                out.write(chunk)
    except HTTPException:
        target.unlink(missing_ok=True)
        raise
    except OSError as exc:
        target.unlink(missing_ok=True)
        raise HTTPException(500, f"Could not store the upload: {exc}")

    if written == 0:
        target.unlink(missing_ok=True)
        raise HTTPException(400, "The uploaded file is empty.")

    duration = probe_duration(target)
    if duration is None:
        target.unlink(missing_ok=True)
        raise HTTPException(400, "Could not read this file as audio or video.")
    if duration > MAX_DURATION_SEC:
        target.unlink(missing_ok=True)
        raise HTTPException(
            413, f"Media is {int(duration)}s long; the limit is {MAX_DURATION_SEC}s."
        )

    # Only now, with the real duration known, claim global budget.
    try:
        _reserve_budget(duration)
    except HTTPException:
        target.unlink(missing_ok=True)
        raise

    job = Job(
        id=job_id,
        filename=(file.filename or target.name)[:200],
        language=language,
        model=chosen,
        duration=duration,
        source_path=str(target),
        terms=request_terms,
    )
    with _jobs_lock:
        _jobs[job_id] = job
    _queue_order.append(job_id)
    _work_queue.put(job_id)

    return {
        "job_id": job_id,
        "duration": duration,
        "model": chosen,
        "language": language,
        "queue_position": len(_queue_order),
    }


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "Unknown or expired job.")
        position = 0
        if job.status == "queued":
            queued = sorted(
                (j for j in _jobs.values() if j.status == "queued"),
                key=lambda j: j.created_at,
            )
            position = next((i + 1 for i, j in enumerate(queued) if j.id == job_id), 0)
        snapshot = {
            "job_id": job.id,
            "status": job.status,
            "stage": job.stage,
            "progress": job.progress,
            "duration": job.duration,
            "filename": job.filename,
            "model": job.model,
            "language": job.language,
            "queue_position": position,
            "cue_count": job.cue_count,
            "error": job.error,
            "srt_ready": job.srt is not None,
        }
    snapshot["eta_seconds"] = _estimate_wait(job_id) if snapshot["status"] != "done" else 0
    return snapshot


@app.get("/api/jobs/{job_id}/srt")
def job_srt(job_id: str, download: bool = False):
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "Unknown or expired job.")
        if job.status == "error":
            raise HTTPException(409, job.error or "Transcription failed.")
        if job.srt is None:
            raise HTTPException(409, f"Not ready yet (status: {job.status}).")
        srt, filename, lang = job.srt, job.filename, job.language

    headers = {}
    if download:
        # Never interpolate a client-supplied name into a header unsanitised:
        # a quote or newline would let the caller inject headers.
        stem = re.sub(r"[^A-Za-z0-9_.-]", "_", Path(filename).stem)[:80] or "subtitles"
        headers["Content-Disposition"] = f'attachment; filename="{stem}_{lang}.srt"'
    return PlainTextResponse(srt, media_type="text/plain; charset=utf-8", headers=headers)


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str):
    with _jobs_lock:
        job = _jobs.pop(job_id, None)
    if job is None:
        raise HTTPException(404, "Unknown or expired job.")
    if job.source_path:
        Path(job.source_path).unlink(missing_ok=True)
    return {"deleted": job_id}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        limit_concurrency=int(os.getenv("LIMIT_CONCURRENCY", "50")),
    )
