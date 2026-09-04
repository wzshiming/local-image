import pytest
import torch

from flux_server.config import Settings
from flux_server.device import (
    Placement,
    apply_placement,
    empty_cache,
    resolve_device,
    resolve_dtype,
    resolve_offload,
    resolve_placement,
)

GIB = 1024**3


def _availability(monkeypatch, *, cuda: bool, mps: bool, bf16: bool = True):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: bf16)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (8, 0))


def test_auto_prefers_cuda(monkeypatch):
    _availability(monkeypatch, cuda=True, mps=True)
    assert resolve_device("auto") == "cuda"


def test_auto_falls_back_to_mps(monkeypatch):
    _availability(monkeypatch, cuda=False, mps=True)
    assert resolve_device("auto") == "mps"


def test_auto_falls_back_to_cpu(monkeypatch):
    _availability(monkeypatch, cuda=False, mps=False)
    assert resolve_device("auto") == "cpu"


def test_explicit_unavailable_raises(monkeypatch):
    _availability(monkeypatch, cuda=False, mps=False)
    with pytest.raises(RuntimeError, match="CUDA"):
        resolve_device("cuda")
    with pytest.raises(RuntimeError, match="MPS"):
        resolve_device("mps")


def test_explicit_cpu_always_ok(monkeypatch):
    _availability(monkeypatch, cuda=False, mps=False)
    assert resolve_device("cpu") == "cpu"


def test_dtype_matrix(monkeypatch):
    _availability(monkeypatch, cuda=True, mps=False, bf16=True)
    assert resolve_dtype("cuda") == torch.bfloat16
    _availability(monkeypatch, cuda=True, mps=False, bf16=False)
    assert resolve_dtype("cuda") == torch.float16
    assert resolve_dtype("mps") == torch.bfloat16
    assert resolve_dtype("cpu") == torch.float32
    assert resolve_dtype("cpu", "float16") == torch.float16
    assert resolve_dtype("mps", "float32") == torch.float32


def test_dtype_uses_float16_on_pre_ampere_cuda(monkeypatch):
    _availability(monkeypatch, cuda=True, mps=False, bf16=True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (7, 5))
    assert resolve_dtype("cuda") == torch.float16


def test_offload_matrix():
    assert resolve_offload("cuda", total_memory_bytes=24 * GIB) == "none"
    assert resolve_offload("cuda", total_memory_bytes=22 * GIB) == "none"
    assert resolve_offload("cuda", total_memory_bytes=16 * GIB) == "model"
    assert resolve_offload("cuda", total_memory_bytes=8 * GIB) == "sequential"
    assert resolve_offload("mps") == "none"
    assert resolve_offload("cpu") == "none"
    assert resolve_offload("cpu", "model") == "none"
    assert resolve_offload("cpu", "sequential") == "none"
    assert resolve_offload("mps", "sequential") == "sequential"
    assert resolve_offload("cuda", "model", total_memory_bytes=80 * GIB) == "model"


def test_resolve_placement(monkeypatch):
    _availability(monkeypatch, cuda=False, mps=True)
    placement = resolve_placement(Settings(_env_file=None))
    assert placement == Placement(device="mps", dtype=torch.bfloat16, offload="none")
    assert placement.dtype_name == "bfloat16"


class FakePipe:
    def __init__(self):
        self.calls: list[tuple[str, str]] = []

    def to(self, device):
        self.calls.append(("to", device))
        return self

    def enable_model_cpu_offload(self, device=None):
        self.calls.append(("model", device))

    def enable_sequential_cpu_offload(self, device=None):
        self.calls.append(("sequential", device))


@pytest.mark.parametrize(
    ("offload", "expected"),
    [("none", "to"), ("model", "model"), ("sequential", "sequential")],
)
def test_apply_placement(offload, expected):
    pipe = FakePipe()
    apply_placement(pipe, Placement(device="cuda", dtype=torch.bfloat16, offload=offload))
    assert pipe.calls == [(expected, "cuda")]


def test_empty_cache_cpu_noop():
    empty_cache("cpu")
