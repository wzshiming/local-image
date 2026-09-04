import asyncio
import logging
import re
import threading
import time
import uuid
from collections.abc import Callable, Sequence
from contextlib import asynccontextmanager
from typing import Any

import PIL.Image
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException as StarletteHTTPException

from flux_server import __version__
from flux_server.config import Settings
from flux_server.engine import (
    EngineProtocol,
    GenerationCancelled,
    GenerationJob,
    GenerationResult,
)
from flux_server.imaging import (
    ImageInputError,
    encode_image,
    load_mask,
    load_upload,
    parse_size,
)
from flux_server.schemas import (
    ErrorDetail,
    ErrorResponse,
    ImageData,
    ImageEditRequest,
    ImageGenerationRequest,
    ImageRequestBase,
    ImagesResponse,
    ModelCard,
    ModelList,
)

log = logging.getLogger("flux_server.api")

_ECHOED_QUALITIES = {"low", "medium", "high"}
REQUEST_ID_HEADER = "X-Request-Id"
MAX_REQUEST_ID_LENGTH = 128
_REQUEST_ID_RE = re.compile(rf"^[A-Za-z0-9._-]{{1,{MAX_REQUEST_ID_LENGTH}}}$")
DISCONNECT_POLL_SECONDS = 0.5


class APIError(Exception):
    def __init__(
        self,
        status_code: int,
        message: str,
        param: str | None = None,
        code: str | None = None,
        error_type: str = "invalid_request_error",
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.param = param
        self.code = code
        self.error_type = error_type


def _error_response(
    status_code: int,
    message: str,
    param: str | None = None,
    code: str | None = None,
    error_type: str = "invalid_request_error",
) -> JSONResponse:
    body = ErrorResponse(
        error=ErrorDetail(message=message, type=error_type, param=param, code=code)
    )
    return JSONResponse(status_code=status_code, content=body.model_dump())


def _api_error_from_validation(errors: Sequence[Any]) -> APIError:
    if not errors:
        return APIError(400, "Invalid request", code="invalid_value")
    first = errors[0]
    loc = [part for part in first.get("loc", ()) if isinstance(part, str) and part != "body"]
    param = loc[-1] if loc else None
    msg = str(first.get("msg", "Invalid value"))
    message = f"{param}: {msg}" if param else msg
    return APIError(400, message, param=param, code="invalid_value")


class JobRegistry:
    """In-flight generations by request id, so they can be cancelled."""

    def __init__(self) -> None:
        self._events: dict[str, threading.Event] = {}
        self._lock = threading.Lock()

    def register(self, request_id: str, limit: int | None = None) -> threading.Event:
        with self._lock:
            if request_id in self._events:
                raise APIError(
                    409,
                    f"A request with {REQUEST_ID_HEADER} {request_id!r} is already in flight",
                    param=REQUEST_ID_HEADER,
                    code="duplicate_request_id",
                )
            if limit is not None and len(self._events) >= limit:
                raise APIError(
                    503,
                    f"Too many in-flight requests (max {limit}); retry later",
                    code="overloaded",
                    error_type="server_error",
                )
            event = threading.Event()
            self._events[request_id] = event
            return event

    def cancel(self, request_id: str) -> bool:
        with self._lock:
            event = self._events.get(request_id)
        if event is None:
            return False
        event.set()
        return True

    def unregister(self, request_id: str) -> None:
        with self._lock:
            self._events.pop(request_id, None)

    def __len__(self) -> int:
        with self._lock:
            return len(self._events)


async def run_cancellable(
    request: Request,
    fn: Callable[[], Any],
    cancel: threading.Event,
    poll_seconds: float = DISCONNECT_POLL_SECONDS,
) -> Any:
    """Run fn in a worker thread; set `cancel` if the client disconnects meanwhile."""
    task = asyncio.ensure_future(asyncio.to_thread(fn))
    try:
        while True:
            done, _ = await asyncio.wait({task}, timeout=poll_seconds)
            if done:
                return task.result()
            if not cancel.is_set() and await request.is_disconnected():
                log.info("client disconnected on %s; cancelling generation", request.url.path)
                cancel.set()
    except asyncio.CancelledError:
        cancel.set()
        raise


def _request_id(request: Request) -> str:
    value = request.headers.get(REQUEST_ID_HEADER, "").strip()
    if not value:
        return uuid.uuid4().hex
    if not _REQUEST_ID_RE.match(value):
        raise APIError(
            400,
            f"{REQUEST_ID_HEADER} must be 1-{MAX_REQUEST_ID_LENGTH} characters from [A-Za-z0-9._-]",
            param=REQUEST_ID_HEADER,
            code="invalid_value",
        )
    return value


async def _load_image_file(
    file: UploadFile,
    loader: Callable[[bytes, str | None], PIL.Image.Image],
    settings: Settings,
    param: str,
) -> PIL.Image.Image:
    data = await file.read()
    if len(data) > settings.max_upload_bytes:
        raise APIError(
            413,
            f"{param} file exceeds the maximum of {settings.max_upload_mb} MB",
            param=param,
            code="file_too_large",
        )
    try:
        return await asyncio.to_thread(loader, data, file.filename)
    except ImageInputError as exc:
        raise APIError(400, str(exc), param=exc.param, code=exc.code) from exc


async def _parse_edit_form(
    request: Request, settings: Settings
) -> tuple[ImageEditRequest, list[PIL.Image.Image], PIL.Image.Image | None]:
    """Parse and validate the multipart body of /v1/images/edits."""
    content_type = request.headers.get("content-type", "")
    if not content_type.lower().startswith("multipart/form-data"):
        raise APIError(400, "Expected multipart/form-data", code="invalid_content_type")
    try:
        form = await request.form()
    except Exception as exc:
        raise APIError(400, "Expected multipart/form-data", code="invalid_content_type") from exc

    # multi_items() keeps wire order even when `image` and `image[]` are mixed.
    files = [
        value
        for key, value in form.multi_items()
        if key in ("image", "image[]") and isinstance(value, UploadFile)
    ]
    if not files:
        raise APIError(
            400, "At least one image file is required", param="image", code="missing_image"
        )
    if len(files) > settings.max_ref_images:
        raise APIError(
            400,
            f"Too many reference images: {len(files)} > {settings.max_ref_images}",
            param="image",
            code="too_many_images",
        )
    refs = [await _load_image_file(f, load_upload, settings, "image") for f in files]

    mask: PIL.Image.Image | None = None
    mask_file = form.get("mask")
    if isinstance(mask_file, UploadFile):
        mask = await _load_image_file(mask_file, load_mask, settings, "mask")
    elif mask_file is not None:
        raise APIError(400, "mask must be an image file", param="mask", code="invalid_mask")

    fields = {key: value for key, value in form.items() if isinstance(value, str)}
    try:
        body = ImageEditRequest.model_validate(fields)
    except ValidationError as exc:
        raise _api_error_from_validation(exc.errors()) from exc

    if mask is not None or body.strength is not None:
        if body.size and body.size.strip().lower() != "auto":
            raise APIError(
                400,
                "size must be 'auto' when mask or strength is used; the output follows the "
                "first image",
                param="size",
                code="unsupported_parameter",
            )
        if len(refs) > 2:
            raise APIError(
                400,
                "mask/strength edits accept the source image plus at most one reference image",
                param="image",
                code="too_many_images",
            )
    return body, refs, mask


def _validate_generation(
    req: ImageRequestBase, settings: Settings
) -> tuple[int | None, int | None, int]:
    """Reject unsupported OpenAI parameters and resolve size/steps."""
    if req.response_format == "url":
        raise APIError(
            400,
            "response_format 'url' is not supported; only b64_json is supported",
            param="response_format",
            code="unsupported_parameter",
        )
    if req.stream:
        raise APIError(400, "stream is not supported", param="stream", code="unsupported_parameter")
    if req.background and req.background.lower() == "transparent":
        raise APIError(
            400,
            "background 'transparent' is not supported",
            param="background",
            code="unsupported_parameter",
        )
    if req.n > settings.max_n:
        raise APIError(
            400, f"n must be between 1 and {settings.max_n}", param="n", code="invalid_value"
        )
    try:
        width, height = parse_size(req.size, settings.max_pixels)
    except ImageInputError as exc:
        raise APIError(400, str(exc), param=exc.param, code=exc.code) from exc
    steps = req.steps if req.steps is not None else settings.steps_for_quality(req.quality)
    return width, height, steps


def create_app(settings: Settings | None = None, engine: EngineProtocol | None = None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.engine is None:
            from flux_server.engine import Engine

            log.info("loading model %s ...", settings.model)
            app.state.engine = await asyncio.to_thread(Engine.from_settings, settings)
            log.info("model ready: %s", app.state.engine.info())
        yield

    app = FastAPI(title="flux-server", version=__version__, lifespan=lifespan)
    app.state.settings = settings
    app.state.engine = engine
    app.state.started_at = int(time.time())
    app.state.jobs = JobRegistry()

    @app.middleware("http")
    async def echo_request_id(request: Request, call_next):
        """Echo a valid client-supplied X-Request-Id on every response, errors included."""
        response = await call_next(request)
        incoming = request.headers.get(REQUEST_ID_HEADER, "").strip()
        if incoming and _REQUEST_ID_RE.match(incoming):
            if REQUEST_ID_HEADER not in response.headers:
                response.headers[REQUEST_ID_HEADER] = incoming
        return response

    @app.exception_handler(APIError)
    async def _handle_api_error(request: Request, exc: APIError) -> JSONResponse:
        return _error_response(exc.status_code, exc.message, exc.param, exc.code, exc.error_type)

    @app.exception_handler(RequestValidationError)
    async def _handle_validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        err = _api_error_from_validation(exc.errors())
        return _error_response(err.status_code, err.message, err.param, err.code)

    @app.exception_handler(StarletteHTTPException)
    async def _handle_http(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = "not_found" if exc.status_code == 404 else None
        return _error_response(exc.status_code, str(exc.detail), code=code)

    @app.exception_handler(Exception)
    async def _handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled error on %s %s", request.method, request.url.path)
        return _error_response(
            500, f"Internal server error: {type(exc).__name__}", error_type="server_error"
        )

    @app.get("/health")
    async def health(request: Request) -> JSONResponse:
        current: EngineProtocol | None = request.app.state.engine
        if current is None:
            return JSONResponse(status_code=503, content={"status": "loading", "ready": False})
        uptime = int(time.time()) - request.app.state.started_at
        return JSONResponse(
            content={
                "status": "ok",
                "ready": True,
                "uptime_seconds": uptime,
                "in_flight": len(request.app.state.jobs),
                **current.info(),
            }
        )

    def _model_card() -> ModelCard:
        return ModelCard(id=settings.model_id, created=app.state.started_at)

    @app.get("/v1/models", response_model=ModelList)
    async def list_models() -> ModelList:
        return ModelList(data=[_model_card()])

    @app.get("/v1/models/{model_id}", response_model=ModelCard)
    async def get_model(model_id: str) -> ModelCard:
        if model_id != settings.model_id:
            raise APIError(
                404, f"The model '{model_id}' does not exist", param="model", code="model_not_found"
            )
        return _model_card()

    @app.post("/v1/images/generations", response_model=ImagesResponse)
    async def generations(body: ImageGenerationRequest, request: Request) -> JSONResponse:
        return await _run(request, body, images=None)

    @app.post("/v1/images/edits", response_model=ImagesResponse)
    async def edits(request: Request) -> JSONResponse:
        body, refs, mask = await _parse_edit_form(request, settings)
        return await _run(request, body, images=refs, mask=mask, strength=body.strength)

    async def _run(
        request: Request,
        req: ImageRequestBase,
        images: list[PIL.Image.Image] | None,
        mask: PIL.Image.Image | None = None,
        strength: float | None = None,
    ) -> JSONResponse:
        current: EngineProtocol | None = request.app.state.engine
        if current is None:
            raise APIError(
                503, "Model is still loading", code="model_loading", error_type="server_error"
            )
        width, height, steps = _validate_generation(req, settings)
        request_id = _request_id(request)
        registry: JobRegistry = request.app.state.jobs
        cancel = registry.register(request_id, limit=settings.max_in_flight)
        job = GenerationJob(
            prompt=req.prompt,
            images=images,
            width=width,
            height=height,
            steps=steps,
            seed=req.seed,
            n=req.n,
            cancel=cancel,
            mask=mask,
            strength=strength,
        )

        def generate_and_encode() -> tuple[GenerationResult, list[ImageData]]:
            result = current.generate(job)
            data = [
                ImageData(
                    b64_json=encode_image(g.image, req.output_format, req.output_compression),
                    seed=g.seed,
                )
                for g in result.images
            ]
            return result, data

        try:
            result, data = await run_cancellable(request, generate_and_encode, cancel)
        except GenerationCancelled as exc:
            if await request.is_disconnected():
                raise APIError(
                    499, "Generation was cancelled: client disconnected", code="cancelled"
                ) from exc
            raise APIError(409, "Generation was cancelled", code="cancelled") from exc
        except ValueError as exc:
            raise APIError(400, str(exc), code="invalid_value") from exc
        finally:
            registry.unregister(request_id)

        quality = req.quality.lower() if req.quality else None
        resp = ImagesResponse(
            created=int(time.time()),
            data=data,
            output_format=req.output_format,
            size=f"{result.width}x{result.height}",
            quality=quality if quality in _ECHOED_QUALITIES else None,
        )
        log.info(
            "%s prompt=%r size=%s steps=%d n=%d seeds=%s refs=%d elapsed=%.2fs",
            request.url.path,
            req.prompt[:60],
            resp.size,
            result.steps,
            req.n,
            [g.seed for g in result.images],
            len(images or []),
            result.elapsed_seconds,
        )
        return JSONResponse(
            content=resp.model_dump(),
            headers={
                "X-Generation-Seconds": f"{result.elapsed_seconds:.3f}",
                REQUEST_ID_HEADER: request_id,
            },
        )

    @app.post("/v1/images/{request_id}/cancel")
    async def cancel_generation(request_id: str, request: Request) -> JSONResponse:
        if not request.app.state.jobs.cancel(request_id):
            raise APIError(
                404,
                f"No in-flight generation with {REQUEST_ID_HEADER} {request_id!r}",
                param="request_id",
                code="not_found",
            )
        log.info("cancel requested for %s", request_id)
        return JSONResponse(content={"request_id": request_id, "cancelled": True})

    if settings.ui:
        from flux_server.ui import mount_ui  # lazy: keep gradio out of API-only processes

        mount_ui(app, settings)

    return app
