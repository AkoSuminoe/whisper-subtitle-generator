"""Tests for the HTTP API: the queue, the limits and the captcha."""

import json
import time


def upload(client, audio, ip="10.0.0.1", token="", name="clip.wav", content=None, **form):
    return client.post(
        "/api/transcribe",
        files={"file": (name, content if content is not None else audio, "audio/wav")},
        data={"language": "tr", "turnstile_token": token, **form},
        headers={"X-Forwarded-For": ip},
    )


def wait_for(client, job_id, timeout=15):
    deadline = time.time() + timeout
    status = {}
    while time.time() < deadline:
        status = client.get(f"/api/jobs/{job_id}").json()
        if status["status"] in ("done", "error"):
            return status
        time.sleep(0.05)
    return status


# --- Happy path ----------------------------------------------------------------------

def test_health_reports_configuration(make_server, audio_bytes):
    client, _ = make_server(MAX_UPLOAD_MB=42)
    with client:
        body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["limits"]["max_upload_mb"] == 42
    assert body["captcha_required"] is False
    assert "budget" in body


def test_transcribe_returns_cleaned_srt(make_server, audio_bytes):
    client, _ = make_server()
    with client:
        response = upload(client, audio_bytes)
        assert response.status_code == 200
        job_id = response.json()["job_id"]

        status = wait_for(client, job_id)
        assert status["status"] == "done", status.get("error")

        srt = client.get(f"/api/jobs/{job_id}/srt").text

    # "Eee" is dropped and the stutter collapsed, so the SRT reflects the pipeline.
    assert "Eee" not in srt
    assert "ben buradayım" in srt
    assert " --> " in srt


def test_download_disposition_is_sanitised(make_server, audio_bytes):
    """A crafted filename must not be able to inject response headers."""
    client, _ = make_server()
    hostile = 'evil";\r\nX-Injected: pwned\r\nZ="a.wav'
    with client:
        job_id = upload(client, audio_bytes, name=hostile).json()["job_id"]
        wait_for(client, job_id)
        response = client.get(f"/api/jobs/{job_id}/srt?download=true")

    assert "x-injected" not in {k.lower() for k in response.headers}
    disposition = response.headers.get("content-disposition", "")
    assert "\r" not in disposition and "\n" not in disposition


def test_job_can_be_deleted(make_server, audio_bytes):
    client, _ = make_server()
    with client:
        job_id = upload(client, audio_bytes).json()["job_id"]
        wait_for(client, job_id)
        assert client.delete(f"/api/jobs/{job_id}").status_code == 200
        assert client.get(f"/api/jobs/{job_id}").status_code == 404


def test_unknown_job_is_404(make_server):
    client, _ = make_server()
    with client:
        assert client.get("/api/jobs/nope").status_code == 404
        assert client.get("/api/jobs/nope/srt").status_code == 404
        assert client.delete("/api/jobs/nope").status_code == 404


# --- Input validation ----------------------------------------------------------------

def test_unsupported_extension_is_rejected(make_server, audio_bytes):
    client, _ = make_server()
    with client:
        assert upload(client, audio_bytes, name="notes.txt").status_code == 400


def test_unsupported_language_is_rejected(make_server, audio_bytes):
    client, _ = make_server()
    with client:
        assert upload(client, audio_bytes, language="de").status_code == 400


def test_empty_upload_is_rejected(make_server, audio_bytes):
    client, _ = make_server()
    with client:
        assert upload(client, audio_bytes, content=b"").status_code == 400


def test_oversized_upload_is_rejected(make_server, audio_bytes):
    client, _ = make_server(MAX_UPLOAD_MB=1)
    with client:
        response = upload(client, audio_bytes, content=b"\x00" * (2 * 1024 * 1024))
    assert response.status_code == 413


def test_over_long_media_is_rejected(make_server, audio_bytes):
    client, _ = make_server(duration=5000.0, MAX_DURATION_SEC=600)
    with client:
        assert upload(client, audio_bytes).status_code == 413


def test_unreadable_media_is_rejected(make_server, audio_bytes, monkeypatch):
    client, main = make_server()
    monkeypatch.setattr(main, "probe_duration", lambda path: None)
    with client:
        assert upload(client, audio_bytes).status_code == 400


def test_model_override_is_ignored_unless_enabled(make_server, audio_bytes):
    client, _ = make_server(WHISPER_MODEL="medium", ALLOW_MODEL_OVERRIDE="false")
    with client:
        body = upload(client, audio_bytes, model="large-v3").json()
    assert body["model"] == "medium"


# --- Limits --------------------------------------------------------------------------

def test_per_client_rate_limit(make_server, audio_bytes):
    client, _ = make_server(RATE_LIMIT_PER_HOUR=2)
    with client:
        codes = [upload(client, audio_bytes, ip="10.0.0.9").status_code for _ in range(3)]
    assert codes[:2] == [200, 200] and codes[2] == 429


def test_global_cap_holds_under_ip_rotation(make_server, audio_bytes):
    """Per-IP limits are defeated by rotation; the global budget is not."""
    client, _ = make_server(RATE_LIMIT_PER_HOUR=100, GLOBAL_JOBS_PER_HOUR=2)
    with client:
        accepted = [upload(client, audio_bytes, ip=f"203.0.113.{i}").status_code
                    for i in range(2)]
        rotated = [upload(client, audio_bytes, ip=f"198.51.100.{i}").status_code
                   for i in range(4)]
    assert accepted == [200, 200]
    assert rotated == [429, 429, 429, 429]


def test_daily_audio_budget(make_server, audio_bytes):
    client, _ = make_server(duration=600.0, GLOBAL_AUDIO_MINUTES_PER_DAY=15)
    with client:
        first = upload(client, audio_bytes, ip="10.1.0.1").status_code
        second = upload(client, audio_bytes, ip="10.1.0.2").status_code
    assert first == 200 and second == 429  # 10 + 10 minutes exceeds 15


def test_budget_survives_a_restart(make_server, audio_bytes, tmp_path):
    client, main = make_server(GLOBAL_JOBS_PER_HOUR=2)
    with client:
        for i in range(2):
            upload(client, audio_bytes, ip=f"10.2.0.{i}")

    saved = json.loads((tmp_path / "budget.json").read_text(encoding="utf-8"))
    assert len(saved["jobs"]) == 2

    # A fresh process reads the file back, so restarting cannot reset the cap.
    client2, _ = make_server(GLOBAL_JOBS_PER_HOUR=2)
    with client2:
        assert client2.get("/api/health").json()["budget"]["jobs_this_hour"] == 2
        assert upload(client2, audio_bytes, ip="10.2.0.99").status_code == 429


def test_queue_depth_cap(make_server, audio_bytes):
    client, _ = make_server(MAX_QUEUE_DEPTH=0)
    with client:
        assert upload(client, audio_bytes).status_code == 503


# --- Captcha and API key -------------------------------------------------------------

def test_captcha_required_when_secret_is_set(make_server, audio_bytes):
    client, _ = make_server(TURNSTILE_SECRET="a-secret")
    with client:
        assert client.get("/api/health").json()["captcha_required"] is True
        assert upload(client, audio_bytes).status_code == 403


def test_captcha_failure_is_closed_when_cloudflare_is_unreachable(
    make_server, audio_bytes, monkeypatch
):
    """An unverifiable request must never reach the GPU."""
    import httpx

    client, main = make_server(TURNSTILE_SECRET="a-secret")

    async def boom(*args, **kwargs):
        raise httpx.ConnectError("no network")

    monkeypatch.setattr(httpx.AsyncClient, "post", boom)
    with client:
        assert upload(client, audio_bytes, token="anything").status_code == 503


def test_api_key_is_enforced(make_server, audio_bytes):
    client, _ = make_server(API_KEY="letmein")
    with client:
        assert upload(client, audio_bytes).status_code == 401
        response = client.post(
            "/api/transcribe",
            files={"file": ("clip.wav", audio_bytes, "audio/wav")},
            data={"language": "tr", "turnstile_token": ""},
            headers={"X-API-Key": "letmein"},
        )
    assert response.status_code == 200


# --- Queue reporting -----------------------------------------------------------------

def test_srt_is_409_before_the_job_finishes(make_server, audio_bytes, monkeypatch):
    client, main = make_server()

    class SlowModel:
        def transcribe(self, path, **kwargs):
            time.sleep(1.0)
            return {"segments": []}

    monkeypatch.setattr(main, "_load_model", lambda name: SlowModel())
    with client:
        job_id = upload(client, audio_bytes).json()["job_id"]
        assert client.get(f"/api/jobs/{job_id}/srt").status_code == 409
        wait_for(client, job_id)


def test_status_reports_queue_position_and_eta(make_server, audio_bytes, monkeypatch):
    client, main = make_server()

    class SlowModel:
        def transcribe(self, path, **kwargs):
            time.sleep(0.8)
            return {"segments": []}

    monkeypatch.setattr(main, "_load_model", lambda name: SlowModel())
    with client:
        first = upload(client, audio_bytes, ip="10.3.0.1").json()["job_id"]
        second = upload(client, audio_bytes, ip="10.3.0.2").json()["job_id"]
        status = client.get(f"/api/jobs/{second}").json()
        assert status["queue_position"] >= 1
        assert status["eta_seconds"] >= 0
        wait_for(client, first)
        wait_for(client, second)


def test_worker_records_an_error_rather_than_dying(make_server, audio_bytes, monkeypatch):
    client, main = make_server()

    class BrokenModel:
        def transcribe(self, path, **kwargs):
            raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(main, "_load_model", lambda name: BrokenModel())
    with client:
        job_id = upload(client, audio_bytes).json()["job_id"]
        status = wait_for(client, job_id)
        assert status["status"] == "error"
        assert "CUDA out of memory" in status["error"]
        # The service still accepts work afterwards.
        assert client.get("/api/health").json()["status"] == "ok"


# --- Caller-supplied terms dictionary --------------------------------------------------

def test_caller_terms_are_applied_to_that_job(make_server, audio_bytes):
    client, _ = make_server()
    with client:
        job_id = upload(
            client, audio_bytes, terms='{"buradayım": "buradayim"}'
        ).json()["job_id"]
        wait_for(client, job_id)
        srt = client.get(f"/api/jobs/{job_id}/srt").text
    assert "buradayim" in srt


def test_terms_are_dropped_when_the_job_finishes(make_server, audio_bytes):
    """The dictionary must not outlive the request that supplied it."""
    client, main = make_server()
    with client:
        job_id = upload(client, audio_bytes, terms='{"a": "b"}').json()["job_id"]
        wait_for(client, job_id)
        assert main._jobs[job_id].terms is None


def test_terms_are_never_written_to_disk(make_server, audio_bytes, tmp_path):
    client, main = make_server()
    before = {p.name for p in tmp_path.iterdir()}
    with client:
        job_id = upload(
            client, audio_bytes, terms='{"gizli": "secret-marker"}'
        ).json()["job_id"]
        wait_for(client, job_id)

    after = {p.name for p in tmp_path.iterdir()}
    assert after == before or after - before <= {"budget.json"}
    for path in tmp_path.rglob("*"):
        if path.is_file():
            assert "secret-marker" not in path.read_text(encoding="utf-8", errors="ignore")


def test_terms_are_isolated_between_jobs(make_server, audio_bytes):
    """One caller's dictionary must not affect the next caller's transcript."""
    client, _ = make_server()
    with client:
        first = upload(
            client, audio_bytes, ip="10.9.0.1", terms='{"buradayım": "MARKED"}'
        ).json()["job_id"]
        wait_for(client, first)
        assert "MARKED" in client.get(f"/api/jobs/{first}/srt").text

        second = upload(client, audio_bytes, ip="10.9.0.2").json()["job_id"]
        wait_for(client, second)
        assert "MARKED" not in client.get(f"/api/jobs/{second}/srt").text


def test_malformed_terms_are_rejected(make_server, audio_bytes):
    client, _ = make_server()
    with client:
        assert upload(client, audio_bytes, terms="{not json").status_code == 400
        assert upload(client, audio_bytes, terms='["a", "b"]').status_code == 400


def test_oversized_terms_are_rejected(make_server, audio_bytes):
    client, _ = make_server(MAX_TERMS_BYTES=200)
    with client:
        big = json.dumps({f"key{i}": f"value{i}" for i in range(100)})
        assert upload(client, audio_bytes, terms=big).status_code == 413


def test_too_many_terms_are_rejected(make_server, audio_bytes):
    client, _ = make_server(MAX_TERMS_ENTRIES=3, MAX_TERMS_BYTES=100000)
    with client:
        many = json.dumps({f"k{i}": "v" for i in range(10)})
        assert upload(client, audio_bytes, terms=many).status_code == 413


def test_over_long_single_term_is_rejected(make_server, audio_bytes):
    client, _ = make_server(MAX_TERM_LENGTH=10)
    with client:
        long_term = json.dumps({"x" * 50: "y"})
        assert upload(client, audio_bytes, terms=long_term).status_code == 413


def test_empty_terms_field_is_ignored(make_server, audio_bytes):
    client, _ = make_server()
    with client:
        for value in ("", "   ", "{}"):
            response = upload(client, audio_bytes, terms=value, ip=f"10.8.0.{len(value)}")
            assert response.status_code == 200


def test_terms_limits_are_advertised(make_server):
    client, _ = make_server(MAX_TERMS_ENTRIES=42)
    with client:
        assert client.get("/api/health").json()["limits"]["max_terms_entries"] == 42
