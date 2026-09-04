from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from flux_server.config import MAX_STEPS


class ImageRequestBase(BaseModel):
    model_config = ConfigDict(extra="ignore")

    prompt: str = Field(min_length=1)
    model: str | None = None
    n: int = Field(1, ge=1)
    size: str | None = None
    quality: str | None = None
    background: str | None = None
    output_format: Literal["png", "jpeg", "webp"] = "png"
    output_compression: int = Field(100, ge=0, le=100)
    response_format: Literal["url", "b64_json"] | None = None
    stream: bool | None = None
    user: str | None = None
    seed: int | None = Field(None, ge=0, le=2**32 - 1)
    steps: int | None = Field(None, ge=1, le=MAX_STEPS)
    partial_images: int | None = None

    @field_validator("prompt")
    @classmethod
    def _strip_prompt(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("prompt must not be empty")
        return value


class ImageGenerationRequest(ImageRequestBase):
    moderation: str | None = None
    style: str | None = None


class ImageEditRequest(ImageRequestBase):
    input_fidelity: str | None = None
    # Extension: how much of the first image to redraw (1 = start from pure noise).
    strength: float | None = Field(None, gt=0, le=1)


class ImageData(BaseModel):
    b64_json: str
    revised_prompt: str | None = None
    seed: int


class ImagesResponse(BaseModel):
    created: int
    data: list[ImageData]
    output_format: str
    size: str
    quality: str | None = None
    background: str = "opaque"


class ErrorDetail(BaseModel):
    message: str
    type: str = "invalid_request_error"
    param: str | None = None
    code: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorDetail


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "black-forest-labs"


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelCard]
