from dataclasses import dataclass
from typing import Any

import torch

from flux_server.config import Settings

_GIB = 1024**3
_DTYPES: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


@dataclass(frozen=True)
class Placement:
    device: str
    dtype: torch.dtype
    offload: str

    @property
    def dtype_name(self) -> str:
        return str(self.dtype).removeprefix("torch.")


def resolve_device(preference: str = "auto") -> str:
    if preference == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if preference == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("FLUX_DEVICE=cuda requested but CUDA is not available")
    if preference == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("FLUX_DEVICE=mps requested but MPS is not available")
    if preference not in ("cuda", "mps", "cpu"):
        raise ValueError(f"Unknown device preference: {preference!r}")
    return preference


def resolve_dtype(device: str, preference: str = "auto") -> torch.dtype:
    if preference != "auto":
        try:
            return _DTYPES[preference]
        except KeyError:
            raise ValueError(f"Unknown dtype preference: {preference!r}") from None
    if device == "cuda":
        if torch.cuda.get_device_capability()[0] < 8:
            return torch.float16
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device == "mps":
        return torch.bfloat16
    return torch.float32


def resolve_offload(
    device: str, preference: str = "auto", total_memory_bytes: int | None = None
) -> str:
    if preference != "auto":
        if preference not in ("none", "model", "sequential"):
            raise ValueError(f"Unknown offload preference: {preference!r}")
        if device == "cpu":
            return "none"
        return preference
    if device != "cuda":
        return "none"
    if total_memory_bytes is None:
        total_memory_bytes = torch.cuda.get_device_properties(0).total_memory
    if total_memory_bytes >= 20 * _GIB:
        return "none"
    if total_memory_bytes >= 10 * _GIB:
        return "model"
    return "sequential"


def resolve_placement(settings: Settings) -> Placement:
    device = resolve_device(settings.device)
    return Placement(
        device=device,
        dtype=resolve_dtype(device, settings.dtype),
        offload=resolve_offload(device, settings.offload),
    )


def apply_placement(pipe: Any, placement: Placement) -> None:
    if placement.offload == "model":
        pipe.enable_model_cpu_offload(device=placement.device)
    elif placement.offload == "sequential":
        pipe.enable_sequential_cpu_offload(device=placement.device)
    else:
        pipe.to(placement.device)


def empty_cache(device: str) -> None:
    try:
        if device == "cuda":
            torch.cuda.empty_cache()
        elif device == "mps":
            torch.mps.empty_cache()
    except AttributeError:
        pass
