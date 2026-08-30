"""Tests for the backend helpers: device detection, the model cache, bookkeeping.

Torch and Whisper are replaced with stubs, so these run without a GPU.
"""

import sys
import time
import types


class FakeCuda:
    def __init__(self, available=True):
        self._available = available
        self.empty_cache_calls = 0

    def is_available(self):
        return self._available

    def get_device_name(self, index):
        return "NVIDIA GeForce RTX 5080"

    def empty_cache(self):
        self.empty_cache_calls += 1


def stub_torch_and_whisper(monkeypatch, cuda_available=True):
    """Put fake torch/whisper modules where `import torch` will find them."""
    torch = types.SimpleNamespace(cuda=FakeCuda(cuda_available))
    loaded = []

    def load_model(name, device=None):
        loaded.append((name, device))
        return f"model:{name}"

    whisper = types.SimpleNamespace(
        load_model=load_model,
        _MODELS={"large-v3": "https://example.com/models/large-v3.pt"},
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "whisper", whisper)
    return torch, whisper, loaded


# --- Device detection ------------------------------------------------------------------

def test_gpu_is_detected(raw_main, monkeypatch):
    stub_torch_and_whisper(monkeypatch, cuda_available=True)
    raw_main._ensure_backend()
    assert raw_main._backend["device"] == "cuda"
    assert raw_main._backend["gpu_name"] == "NVIDIA GeForce RTX 5080"


def test_cpu_fallback(raw_main, monkeypatch):
    stub_torch_and_whisper(monkeypatch, cuda_available=False)
    raw_main._ensure_backend()
    assert raw_main._backend["device"] == "cpu"
    assert raw_main._backend["gpu_name"] is None


def test_backend_is_imported_only_once(raw_main, monkeypatch):
    stub_torch_and_whisper(monkeypatch, cuda_available=True)
    raw_main._ensure_backend()
    first = raw_main._backend["whisper"]
    stub_torch_and_whisper(monkeypatch, cuda_available=False)  # would flip the device
    raw_main._ensure_backend()
    assert raw_main._backend["whisper"] is first
    assert raw_main._backend["device"] == "cuda"


# --- Model cache -----------------------------------------------------------------------

def test_model_is_reused_between_calls(raw_main, monkeypatch):
    torch, whisper, loaded = stub_torch_and_whisper(monkeypatch)
    raw_main._backend.update({"torch": torch, "whisper": whisper, "device": "cuda"})

    first = raw_main._load_model("large-v3")
    second = raw_main._load_model("large-v3")

    assert first is second
    assert len(loaded) == 1


def test_switching_model_frees_vram_first(raw_main, monkeypatch):
    torch, whisper, loaded = stub_torch_and_whisper(monkeypatch)
    raw_main._backend.update({"torch": torch, "whisper": whisper, "device": "cuda"})

    raw_main._load_model("large-v3")
    raw_main._load_model("medium")

    assert [name for name, _ in loaded] == ["large-v3", "medium"]
    assert torch.cuda.empty_cache_calls >= 1


def test_cpu_does_not_touch_cuda(raw_main, monkeypatch):
    torch, whisper, _loaded = stub_torch_and_whisper(monkeypatch)
    raw_main._backend.update({"torch": torch, "whisper": whisper, "device": "cpu"})
    raw_main._load_model("medium")
    assert torch.cuda.empty_cache_calls == 0


def test_weights_presence_is_detected(raw_main, monkeypatch, tmp_path):
    _torch, whisper, _ = stub_torch_and_whisper(monkeypatch)
    raw_main._backend["whisper"] = whisper
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))

    assert raw_main._model_is_cached("large-v3") is False
    weights = tmp_path / "whisper"
    weights.mkdir(parents=True, exist_ok=True)
    (weights / "large-v3.pt").write_bytes(b"x")
    assert raw_main._model_is_cached("large-v3") is True


def test_unknown_model_name_assumes_cached(raw_main, monkeypatch):
    """An unrecognised name must not claim a multi-GB download is pending."""
    _torch, whisper, _ = stub_torch_and_whisper(monkeypatch)
    raw_main._backend["whisper"] = whisper
    assert raw_main._model_is_cached("no-such-model") is True


# --- Budget bookkeeping -----------------------------------------------------------------

def test_entries_older_than_a_day_are_pruned(raw_main):
    now = time.time()
    raw_main._budget["jobs"] = [now - 90000, now - 100]
    raw_main._budget["audio"] = [[now - 90000, 600], [now - 100, 60]]
    snapshot = raw_main._budget_snapshot()
    assert snapshot["jobs_today"] == 1
    assert snapshot["audio_minutes_today"] == 1.0


def test_corrupt_budget_file_starts_empty(raw_main):
    raw_main.BUDGET_FILE.write_text("{ not json", encoding="utf-8")
    raw_main._load_budget()
    assert raw_main._budget == {"jobs": [], "audio": []}


def test_budget_round_trips_through_the_file(raw_main):
    raw_main._budget = {"jobs": [1.0, 2.0], "audio": [[1.0, 30.0]]}
    raw_main._save_budget()
    raw_main._budget = {"jobs": [], "audio": []}
    raw_main._load_budget()
    assert raw_main._budget["jobs"] == [1.0, 2.0]
    assert raw_main._budget["audio"] == [[1.0, 30.0]]


# --- Client identification ---------------------------------------------------------------

def test_client_key_prefers_the_cloudflare_header(raw_main):
    class Req:
        def __init__(self, headers):
            self.headers = headers
            self.client = types.SimpleNamespace(host="127.0.0.1")

    assert raw_main._client_key(Req({"cf-connecting-ip": "1.2.3.4"})) == "1.2.3.4"
    assert raw_main._client_key(Req({"x-forwarded-for": "5.6.7.8, 9.9.9.9"})) == "5.6.7.8"
    assert raw_main._client_key(Req({})) == "127.0.0.1"
