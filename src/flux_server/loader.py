import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from flux_server.config import Settings
from flux_server.device import Placement, apply_placement, resolve_placement

log = logging.getLogger("flux_server.loader")

DOWNLOAD_HINT = (
    "Run `make download` (hf download black-forest-labs/FLUX.2-klein-4B) to fetch the weights."
)


@dataclass
class LoadedPipeline:
    pipe: Any
    snapshot_dir: Path
    weights_source: str
    placement: Placement


def resolve_snapshot_dir(settings: Settings) -> Path:
    if settings.model_path:
        path = Path(settings.model_path).expanduser()
        if not path.is_dir():
            raise FileNotFoundError(f"FLUX_MODEL_PATH {path} does not exist. {DOWNLOAD_HINT}")
        return path
    from huggingface_hub import snapshot_download

    try:
        return Path(snapshot_download(settings.model, local_files_only=settings.local_files_only))
    except Exception as exc:
        raise FileNotFoundError(
            f"Model {settings.model!r} not found in the local cache ({exc}). {DOWNLOAD_HINT}"
        ) from exc


def has_diffusers_transformer(snapshot: Path) -> bool:
    return any(
        p.exists() and p.stat().st_size > 0
        for p in (snapshot / "transformer").glob("diffusion_pytorch_model*.safetensors")
    )


def find_single_file(snapshot: Path) -> Path | None:
    candidates = [p for p in snapshot.glob("*.safetensors") if p.exists() and p.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_size)


def select_weights_source(snapshot: Path, preference: str = "auto") -> str:
    has_diffusers = has_diffusers_transformer(snapshot)
    single = find_single_file(snapshot)
    if preference == "auto":
        if has_diffusers:
            return "diffusers"
        if single is not None:
            return "single_file"
        raise FileNotFoundError(f"No transformer weights found in {snapshot}. {DOWNLOAD_HINT}")
    if preference == "diffusers":
        if not has_diffusers:
            raise FileNotFoundError(
                f"transformer/diffusion_pytorch_model*.safetensors missing in {snapshot}. "
                f"{DOWNLOAD_HINT}"
            )
        return "diffusers"
    if preference == "single_file":
        if single is None:
            raise FileNotFoundError(
                f"No single-file *.safetensors found in {snapshot}. {DOWNLOAD_HINT}"
            )
        return "single_file"
    raise ValueError(f"Unknown weights preference: {preference!r}")


def load_pipeline(settings: Settings, placement: Placement | None = None) -> LoadedPipeline:
    import diffusers

    placement = placement or resolve_placement(settings)
    snapshot = resolve_snapshot_dir(settings)
    source = select_weights_source(snapshot, settings.weights)
    common = {"dtype": placement.dtype, "local_files_only": settings.local_files_only}

    started = time.perf_counter()
    log.info(
        "Loading %s from %s (weights=%s, device=%s, dtype=%s, offload=%s)",
        settings.model,
        snapshot,
        source,
        placement.device,
        placement.dtype_name,
        placement.offload,
    )
    if source == "single_file":
        single = find_single_file(snapshot)
        transformer = diffusers.Flux2Transformer2DModel.from_single_file(
            str(single), config=str(snapshot), subfolder="transformer", **common
        )
        pipe = diffusers.Flux2KleinPipeline.from_pretrained(
            str(snapshot), transformer=transformer, **common
        )
    else:
        pipe = diffusers.Flux2KleinPipeline.from_pretrained(str(snapshot), **common)
    apply_placement(pipe, placement)
    log.info("Pipeline ready in %.1fs", time.perf_counter() - started)
    return LoadedPipeline(
        pipe=pipe, snapshot_dir=snapshot, weights_source=source, placement=placement
    )
