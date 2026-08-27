"""FastAPI service exposing the subtitle pipeline.

One GPU means one job at a time, so work goes through a single worker thread.
Uploads are capped by size and duration and rate limited per client.
"""

import os
import queue
import sys
import tempfile
import threading
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

# subtitle_core lives one level up, beside app.py.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from subtitle_core import (  # noqa: E402
    LANGUAGES,
    MODEL_CHOICES,
    TermReplacer,
    build_srt,
    configure_utf8_console,
    load_terms,
    postprocess_segments,
    probe_duration,
    whisper_cache_dir,
    whisper_progress,
)

configure_utf8_console()

# --- Configuration (environment variables) -----------------------------------------
DEFAULT_MODEL = os.getenv("WHISPER_MODEL", "large-v3")
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "200"))
MAX_DURATION_SEC = int(os.getenv("MAX_DURATION_SEC", "900"))  # 15 minutes
RATE_LIMIT_PER_HOUR = int(os.getenv("RATE_LIMIT_PER_HOUR", "5"))
JOB_TTL_SEC = int(os.getenv("JOB_TTL_SEC", "3600"))
API_KEY = os.getenv("API_KEY", "").strip()
ALLOWED_ORIGINS = [
    o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()
]
ALLOW_MODEL_OVERRIDE = os.getenv("ALLOW_MODEL_OVERRIDE", "false").lower() == "true"
TERMS_FILE = os.getenv("TERMS_FILE", "").strip()

MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
ALLOWED_EXTENSIONS = {
    ".mp4", ".mov", ".mkv", ".avi", ".webm",
    ".mp3", ".wav", ".m4a", ".ogg", ".flac",
}

app = FastAPI(title="Whisper Subtitle Generator API", version="1.0.0")
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
    source_path: Optional[str] = None


_jobs = {}
_jobs_lock = threading.Lock()
_work_queue = queue.Queue()

# Rate limiting: recent request timestamps per client key.
_hits = defaultdict(deque)
_hits_lock = threading.Lock()

# Backend handles, loaded lazily on the worker thread.
_backend = {"torch": None, "whisper": None, "device": "cpu", "gpu_name": None}
_model_cache = {"key": None, "model": None}
_backend_lock = threading.Lock()


def _client_key(request: Request) -> str:
    """Identify a caller for rate limiting, honouring a reverse proxy header."""
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _check_rate_limit(key: str) -> None:
    if RATE_LIMIT_PER_HOUR <= 0:
        return
    now = time.time()
    with _hits_lock:
        hits = _hits[key]
        while hits and now - hits[0] > 3600:
            hits.popleft()
        if len(hits) >= RATE_LIMIT_PER_HOUR:
            retry_in = int(3600 - (now - hits[0]))
            raise HTTPException(
                status_code=429,
                detail=(
                    f"Rate limit reached ({RATE_LIMIT_PER_HOUR}/hour). "
                    f"Try again in {retry_in}s."
                ),
            )
        hits.append(now)


def _require_api_key(provided: Optional[str]) -> None:
    if API_KEY and (provided or "").strip() != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


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
    """True when the weights are already on disk, so no download stage is needed."""
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


def _worker_loop() -> None:
    """Process one job at a time: a single GPU cannot usefully do more."""
    while True:
        job_id = _work_queue.get()
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
    try:
        _update(job_id, status="running", stage="Loading model", progress=0.0)
        _ensure_backend()

        if not _model_is_cached(job.model):
            _update(job_id, stage=f"Downloading {job.model} (first run only)")
        model = _load_model(job.model)

        _update(job_id, stage="Transcribing", progress=0.0)

        def on_progress(fraction: float) -> None:
            _update(job_id, progress=round(float(fraction), 4))

        with whisper_progress(on_progress):
            result = model.transcribe(
                str(source),
                language=job.language,
                fp16=(_backend["device"] == "cuda"),
                verbose=False,
            )

        _update(job_id, stage="Cleaning transcript", progress=1.0)
        terms, _warning = load_terms(Path(TERMS_FILE) if TERMS_FILE else None)
        cues = postprocess_segments(
            result.get("segments", []), TermReplacer(terms, job.language)
        )
        srt = build_srt(cues)

        _update(
            job_id,
            status="done",
            stage="Done",
            progress=1.0,
            srt=srt,
            cue_count=len(cues),
        )
    finally:
        # The upload is only needed during transcription.
        try:
            source.unlink(missing_ok=True)
        except OSError:
            pass
        _update(job_id, source_path=None)


def _cleanup_loop() -> None:
    """Drop finished jobs (and their SRT text) once they age out."""
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


@app.on_event("startup")
def _startup() -> None:
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
        "limits": {
            "max_upload_mb": MAX_UPLOAD_MB,
            "max_duration_sec": MAX_DURATION_SEC,
            "rate_limit_per_hour": RATE_LIMIT_PER_HOUR,
            "api_key_required": bool(API_KEY),
        },
    }


@app.post("/api/transcribe")
async def transcribe(
    request: Request,
    file: UploadFile = File(...),
    language: str = Form("tr"),
    model: str = Form(""),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
):
    _require_api_key(x_api_key)
    _check_rate_limit(_client_key(request))

    if language not in LANGUAGES.values():
        allowed = sorted(set(LANGUAGES.values()))
        raise HTTPException(400, f"Unsupported language. Use one of {allowed}.")

    chosen = (model or DEFAULT_MODEL).strip()
    if not ALLOW_MODEL_OVERRIDE:
        chosen = DEFAULT_MODEL
    elif chosen not in MODEL_CHOICES:
        raise HTTPException(400, f"Unknown model. Use one of {MODEL_CHOICES}.")

    suffix = Path(file.filename or "audio").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise HTTPException(400, f"Unsupported file type. Allowed: {allowed}.")

    # Stream to disk, enforcing the size cap as we go rather than buffering it all.
    upload_dir = Path(tempfile.gettempdir()) / "whisper_api_uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    job_id = uuid.uuid4().hex
    target = upload_dir / f"{job_id}{suffix}"

    written = 0
    try:
        with target.open("wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        413, f"File exceeds the {MAX_UPLOAD_MB} MB limit."
                    )
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

    job = Job(
        id=job_id,
        filename=file.filename or target.name,
        language=language,
        model=chosen,
        duration=duration,
        source_path=str(target),
    )
    with _jobs_lock:
        _jobs[job_id] = job
    _work_queue.put(job_id)

    return {
        "job_id": job_id,
        "duration": duration,
        "model": chosen,
        "language": language,
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
            position = next(
                (i + 1 for i, j in enumerate(queued) if j.id == job_id), 0
            )
        return {
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
        stem = Path(filename).stem or "subtitles"
        headers["Content-Disposition"] = f'attachment; filename="{stem}_{lang}.srt"'
    return PlainTextResponse(
        srt, media_type="text/plain; charset=utf-8", headers=headers
    )


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
    )
