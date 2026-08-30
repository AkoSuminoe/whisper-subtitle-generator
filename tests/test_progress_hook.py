"""Tests for the Whisper progress hook, without needing a GPU or a model.

Whisper exposes no progress callback, so `whisper_progress` swaps the tqdm bar
its transcribe loop drives. These tests stand in for that loop.
"""

import sys

import pytest

import subtitle_core as core

whisper_transcribe = pytest.importorskip(
    "whisper.transcribe", reason="openai-whisper is not installed"
)


def drive_bar(total, steps):
    """Do what whisper/transcribe.py does: construct the bar and update it."""
    module = sys.modules["whisper.transcribe"]
    bar_class = module.tqdm.tqdm
    with bar_class(total=total, unit="frames", disable=False) as bar:
        for step in steps:
            bar.update(step)


def test_progress_is_reported_as_a_fraction():
    seen = []
    with core.whisper_progress(seen.append):
        drive_bar(100, [25, 25, 50])
    assert seen == [0.25, 0.5, 1.0]


def test_progress_never_exceeds_one():
    seen = []
    with core.whisper_progress(seen.append):
        drive_bar(100, [80, 80])
    assert max(seen) == 1.0


def test_the_real_tqdm_is_restored_afterwards():
    import tqdm as real_tqdm

    module = sys.modules["whisper.transcribe"]
    before = module.tqdm
    with core.whisper_progress(lambda f: None):
        assert module.tqdm is not before  # the shim is installed
    assert module.tqdm is before is real_tqdm


def test_the_hook_is_restored_even_if_transcribe_raises():
    module = sys.modules["whisper.transcribe"]
    before = module.tqdm
    with pytest.raises(RuntimeError):
        with core.whisper_progress(lambda f: None):
            raise RuntimeError("transcription blew up")
    assert module.tqdm is before


def test_a_callback_that_raises_propagates():
    """The job watchdog aborts a run by raising from the callback."""

    def watchdog(fraction):
        raise TimeoutError("took too long")

    with pytest.raises(TimeoutError):
        with core.whisper_progress(watchdog):
            drive_bar(100, [50])


def test_shim_tolerates_a_zero_total():
    seen = []
    with core.whisper_progress(seen.append):
        drive_bar(0, [1])
    assert seen == []  # nothing to divide by, so nothing is reported


def test_missing_whisper_degrades_instead_of_failing(monkeypatch):
    """An unknown Whisper build must not break transcription, only progress."""
    import importlib

    def no_whisper(name):
        raise ImportError(name)

    monkeypatch.setattr(importlib, "import_module", no_whisper)
    with core.whisper_progress(lambda f: None):
        pass  # must not raise


def test_module_without_tqdm_degrades(monkeypatch):
    import importlib
    import types

    monkeypatch.setattr(
        importlib, "import_module", lambda name: types.SimpleNamespace()
    )
    with core.whisper_progress(lambda f: None):
        pass


# --- Supporting helpers --------------------------------------------------------------

def test_probe_duration_returns_none_for_a_missing_file(tmp_path):
    assert core.probe_duration(tmp_path / "does-not-exist.mp4") is None


def test_probe_duration_returns_none_for_a_non_media_file(tmp_path):
    junk = tmp_path / "junk.wav"
    junk.write_bytes(b"definitely not audio")
    assert core.probe_duration(junk) is None


def test_whisper_cache_dir_is_under_the_cache_home(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert core.whisper_cache_dir() == str(tmp_path / "whisper")


def test_probe_duration_reads_a_real_file(tmp_path):
    """Runs ffprobe for real. Catches platform bugs a stubbed test cannot.

    This is the test that would have caught `creationflags` being passed on
    non-Windows platforms, where it raises and silently yields None.
    """
    import shutil
    import subprocess

    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("ffmpeg is not installed")

    clip = tmp_path / "tone.wav"
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi",
         "-i", "sine=frequency=440:duration=3", str(clip)],
        check=True,
    )
    assert core.probe_duration(clip) == pytest.approx(3.0, abs=0.2)
