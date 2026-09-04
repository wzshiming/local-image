import logging
import random
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import PIL.Image
import torch

from flux_server.config import Settings
from flux_server.device import Placement, empty_cache
from flux_server.imaging import composite_with_mask

log = logging.getLogger("flux_server.engine")


class GenerationCancelled(Exception):
    """Raised by Engine.generate when the job's cancel event is set."""


@dataclass(frozen=True)
class GenerationJob:
    prompt: str
    images: list[PIL.Image.Image] | None = None
    width: int | None = None
    height: int | None = None
    steps: int = 4
    seed: int | None = None
    n: int = 1
    cancel: threading.Event | None = None  # set() aborts at the next denoising step / image
    mask: PIL.Image.Image | None = None  # 'L', 255 = repaint; applies to images[0]
    strength: float | None = None  # (0, 1]; with mask or strength the job is an inpaint job

    @property
    def inpaint(self) -> bool:
        return self.mask is not None or self.strength is not None


@dataclass(frozen=True)
class GeneratedImage:
    image: PIL.Image.Image
    seed: int


@dataclass(frozen=True)
class GenerationResult:
    images: list[GeneratedImage]
    elapsed_seconds: float
    steps: int
    width: int
    height: int


class EngineProtocol(Protocol):
    model_id: str

    def generate(self, job: GenerationJob) -> GenerationResult: ...

    def info(self) -> dict[str, Any]: ...


def build_inpaint_pipe(pipe: Any) -> Any:
    """Inpaint pipeline over the already loaded components (no extra weights, hooks stay)."""
    from diffusers import Flux2KleinInpaintPipeline

    return Flux2KleinInpaintPipeline(
        scheduler=pipe.scheduler,
        vae=pipe.vae,
        text_encoder=pipe.text_encoder,
        tokenizer=pipe.tokenizer,
        transformer=pipe.transformer,
        is_distilled=bool(getattr(pipe.config, "is_distilled", True)),
    )


class Engine:
    def __init__(
        self,
        pipe: Any,
        settings: Settings,
        placement: Placement,
        weights_source: str,
        snapshot_dir: Path | None = None,
        inpaint_pipe: Any = None,
    ) -> None:
        self.pipe = pipe
        self.settings = settings
        self.placement = placement
        self.weights_source = weights_source
        self.snapshot_dir = snapshot_dir
        self.model_id = settings.model_id
        self._inpaint_pipe = inpaint_pipe
        self._lock = threading.Lock()

    @classmethod
    def from_settings(cls, settings: Settings) -> "Engine":
        from flux_server.loader import load_pipeline

        loaded = load_pipeline(settings)
        engine = cls(
            loaded.pipe, settings, loaded.placement, loaded.weights_source, loaded.snapshot_dir
        )
        if settings.warmup:
            engine.generate(GenerationJob(prompt="warmup", width=256, height=256, steps=1, seed=0))
        return engine

    def generate(self, job: GenerationJob) -> GenerationResult:
        if job.n < 1:
            raise ValueError("n must be >= 1")
        if job.steps < 1:
            raise ValueError("steps must be >= 1")
        if job.inpaint:
            if not job.images:
                raise ValueError("mask/strength require a source image")
            if len(job.images) > 2:
                raise ValueError("mask/strength accept at most one extra reference image")
            if job.strength is not None and not 0 < job.strength <= 1:
                raise ValueError("strength must be in (0, 1]")
        base_seed = job.seed if job.seed is not None else random.SystemRandom().randrange(2**31)
        images: list[GeneratedImage] = []

        def check_cancelled() -> None:
            if job.cancel is not None and job.cancel.is_set():
                raise GenerationCancelled()

        def on_step_end(pipe: Any, step: int, timestep: Any, kwargs: dict) -> dict:
            check_cancelled()
            return {}

        common: dict[str, Any] = {
            "prompt": job.prompt,
            "num_inference_steps": job.steps,
            "guidance_scale": 1.0,
            "num_images_per_prompt": 1,
            "output_type": "pil",
            "callback_on_step_end": on_step_end if job.cancel is not None else None,
        }

        def run_one(generator: torch.Generator) -> PIL.Image.Image:
            if not job.inpaint:
                out = self.pipe(
                    image=job.images or None,
                    height=job.height,
                    width=job.width,
                    generator=generator,
                    **common,
                )
                return out.images[0]
            source, refs = job.images[0], job.images[1:]
            mask = job.mask if job.mask is not None else PIL.Image.new("L", source.size, 255)
            out = self._get_inpaint_pipe()(
                image=source,
                mask_image=mask,
                image_reference=refs[0] if refs else None,
                strength=job.strength if job.strength is not None else 1.0,
                generator=generator,
                **common,
            )
            image = out.images[0]
            return composite_with_mask(image, source, mask) if job.mask is not None else image

        with self._lock, torch.inference_mode():
            started = time.perf_counter()
            try:
                for i in range(job.n):
                    check_cancelled()
                    seed = (base_seed + i) % 2**32
                    generator = torch.Generator("cpu").manual_seed(seed)
                    images.append(GeneratedImage(image=run_one(generator), seed=seed))
            except GenerationCancelled:
                log.info(
                    "cancelled prompt=%r after %d/%d image(s) %.1fs",
                    job.prompt[:60],
                    len(images),
                    job.n,
                    time.perf_counter() - started,
                )
                raise
            finally:
                empty_cache(self.placement.device)
            elapsed = time.perf_counter() - started
        width, height = images[0].image.size
        log.info(
            "generated prompt=%r size=%dx%d steps=%d seeds=%s n=%d refs=%d mask=%s strength=%s "
            "elapsed=%.1fs",
            job.prompt[:60],
            width,
            height,
            job.steps,
            [g.seed for g in images],
            job.n,
            len(job.images or []),
            job.mask is not None,
            job.strength,
            elapsed,
        )
        return GenerationResult(
            images=images, elapsed_seconds=elapsed, steps=job.steps, width=width, height=height
        )

    def _get_inpaint_pipe(self) -> Any:
        if self._inpaint_pipe is None:
            self._inpaint_pipe = build_inpaint_pipe(self.pipe)
        return self._inpaint_pipe

    def info(self) -> dict[str, Any]:
        return {
            "model": self.settings.model,
            "model_id": self.model_id,
            "device": self.placement.device,
            "dtype": self.placement.dtype_name,
            "offload": self.placement.offload,
            "weights_source": self.weights_source,
            "snapshot_dir": str(self.snapshot_dir) if self.snapshot_dir else None,
            "default_steps": self.settings.default_steps,
            "ready": True,
        }
