from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

MAX_STEPS = 50  # hard cap on inference steps, shared by the schema and the quality map


def _default_quality_steps() -> dict[str, int]:
    return {"low": 2, "medium": 4, "high": 8, "auto": 4, "standard": 4, "hd": 8}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="FLUX_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    model: str = "black-forest-labs/FLUX.2-klein-4B"
    model_path: str | None = None
    model_id: str = "flux.2-klein-4b"
    device: Literal["auto", "cuda", "mps", "cpu"] = "auto"
    dtype: Literal["auto", "bfloat16", "float16", "float32"] = "auto"
    offload: Literal["auto", "none", "model", "sequential"] = "auto"
    weights: Literal["auto", "diffusers", "single_file"] = "auto"
    host: str = "127.0.0.1"
    port: int = 8000
    default_steps: int = 4
    quality_steps: dict[str, int] = Field(default_factory=_default_quality_steps)
    max_n: int = 10
    max_pixels: int = 2048 * 2048
    max_ref_images: int = 16
    max_upload_mb: int = 32
    max_in_flight: int = 8
    ui: bool = True
    ui_api_base: str | None = None
    warmup: bool = False
    local_files_only: bool = True

    def steps_for_quality(self, quality: str | None) -> int:
        steps = self.default_steps
        if quality is not None:
            steps = self.quality_steps.get(quality.lower(), self.default_steps)
        return min(max(steps, 1), MAX_STEPS)

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def api_base(self) -> str:
        host = "127.0.0.1" if self.host in ("0.0.0.0", "::", "") else self.host
        if ":" in host:  # bare IPv6 needs brackets in a URL
            host = f"[{host}]"
        base = self.ui_api_base or f"http://{host}:{self.port}/v1"
        return base.rstrip("/")


def get_settings(**overrides) -> Settings:
    return Settings(**overrides)
