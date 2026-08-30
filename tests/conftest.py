"""Shared fixtures.

The server tests run without a GPU, ffmpeg or network: the Whisper backend is
replaced with a stub, so what is under test is the queue, the limits and the
captcha rather than transcription itself.
"""

import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "server"))

FAKE_SEGMENTS = [
    {"start": 0.0, "end": 2.0, "text": "Bu bir test kaydıdır"},
    {"start": 2.0, "end": 3.0, "text": "Eee"},
    {"start": 3.0, "end": 6.0, "text": "ben ben buradayım"},
]


class FakeModel:
    def transcribe(self, path, **kwargs):
        return {"segments": [dict(s) for s in FAKE_SEGMENTS]}


@pytest.fixture
def make_server(tmp_path, monkeypatch):
    """Build a TestClient over a freshly configured server module."""
    from fastapi.testclient import TestClient

    def _factory(duration=30.0, **env):
        settings = {
            "BUDGET_FILE": str(tmp_path / "budget.json"),
            "TURNSTILE_SECRET": "",
            "API_KEY": "",
            "RATE_LIMIT_PER_HOUR": "100",
            "RATE_LIMIT_PER_DAY": "100",
            "GLOBAL_JOBS_PER_HOUR": "100",
            "GLOBAL_JOBS_PER_DAY": "100",
            "GLOBAL_AUDIO_MINUTES_PER_DAY": "1000",
            "MAX_QUEUE_DEPTH": "10",
            "MAX_UPLOAD_MB": "5",
            "MAX_DURATION_SEC": "600",
        }
        settings.update({k: str(v) for k, v in env.items()})
        for key, value in settings.items():
            monkeypatch.setenv(key, value)

        import main

        importlib.reload(main)

        # Stub the parts that would need a GPU or ffmpeg.
        monkeypatch.setattr(main, "probe_duration", lambda path: duration)
        monkeypatch.setattr(main, "_ensure_backend", lambda: main._backend)
        monkeypatch.setattr(main, "_load_model", lambda name: FakeModel())
        monkeypatch.setattr(main, "_model_is_cached", lambda name: True)

        return TestClient(main.app), main

    return _factory


@pytest.fixture
def audio_bytes():
    """Any bytes will do: probe_duration is stubbed, so nothing decodes this."""
    return b"\x00" * 2048


@pytest.fixture
def raw_main(tmp_path, monkeypatch):
    """The server module with nothing stubbed, for unit-testing its helpers."""
    monkeypatch.setenv("BUDGET_FILE", str(tmp_path / "budget.json"))
    monkeypatch.setenv("TURNSTILE_SECRET", "")

    import main

    importlib.reload(main)
    main._backend.update(
        {"torch": None, "whisper": None, "device": "cpu", "gpu_name": None}
    )
    main._model_cache.update({"key": None, "model": None})
    return main
