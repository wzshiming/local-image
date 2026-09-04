import asyncio
from io import BytesIO
from typing import Any

import PIL.Image
import pytest
from fastapi.testclient import TestClient

from flux_server.api import create_app
from flux_server.config import Settings
from flux_server.engine import GeneratedImage, GenerationJob, GenerationResult


class FakeEngine:
    model_id = "flux.2-klein-4b"

    def __init__(self) -> None:
        self.jobs: list[GenerationJob] = []
        self.saw_running_loop: list[bool] = []

    def generate(self, job: GenerationJob) -> GenerationResult:
        self.jobs.append(job)
        try:
            asyncio.get_running_loop()
            self.saw_running_loop.append(True)
        except RuntimeError:
            self.saw_running_loop.append(False)
        if job.width is not None and job.height is not None:
            w, h = job.width, job.height
        elif job.images:
            w, h = job.images[0].size
        else:
            w, h = 1024, 1024
        base = job.seed if job.seed is not None else 12345
        images = [
            GeneratedImage(
                image=PIL.Image.new("RGB", (w, h), (i * 40 % 256, 80, 120)), seed=base + i
            )
            for i in range(job.n)
        ]
        return GenerationResult(
            images=images, elapsed_seconds=0.25, steps=job.steps, width=w, height=h
        )

    def info(self) -> dict[str, Any]:
        return {
            "model": "fake/flux",
            "model_id": self.model_id,
            "device": "cpu",
            "dtype": "float32",
            "offload": "none",
            "weights_source": "fake",
            "snapshot_dir": None,
            "default_steps": 4,
            "ready": True,
        }


def png_bytes(w: int = 64, h: int = 64, color: str | tuple = "red") -> bytes:
    buf = BytesIO()
    PIL.Image.new("RGB", (w, h), color).save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None, ui=False)


@pytest.fixture
def engine() -> FakeEngine:
    return FakeEngine()


@pytest.fixture
def client(settings: Settings, engine: FakeEngine):
    with TestClient(create_app(settings, engine=engine)) as c:
        yield c
