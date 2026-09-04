import base64
import re
from io import BytesIO

import PIL.Image
from PIL import ImageOps, UnidentifiedImageError
from PIL.Image import DecompressionBombError

MIME_TYPES = {"png": "image/png", "jpeg": "image/jpeg", "webp": "image/webp"}

_SIZE_RE = re.compile(r"^\s*(\d{1,5})\s*[xX×]\s*(\d{1,5})\s*$")
_MIN_DIM = 64


class ImageInputError(ValueError):
    def __init__(self, message: str, param: str | None = None, code: str | None = None) -> None:
        super().__init__(message)
        self.param = param
        self.code = code


def _round16(value: int) -> int:
    return max(_MIN_DIM, round(value / 16) * 16)


def parse_size(size: str | None, max_pixels: int) -> tuple[int | None, int | None]:
    if size is None or size.strip() == "" or size.strip().lower() == "auto":
        return None, None
    match = _SIZE_RE.match(size)
    if not match or int(match.group(1)) == 0 or int(match.group(2)) == 0:
        raise ImageInputError(
            f"Invalid size {size!r}: expected 'WIDTHxHEIGHT' (e.g. '1024x1024') or 'auto'",
            param="size",
            code="invalid_size",
        )
    width, height = _round16(int(match.group(1))), _round16(int(match.group(2)))
    if width * height > max_pixels:
        raise ImageInputError(
            f"Size {width}x{height} exceeds the maximum of {max_pixels} pixels",
            param="size",
            code="size_too_large",
        )
    # Ratio is checked on the requested values so exact 3:1 inputs survive rounding to 16.
    ratio = int(match.group(1)) / int(match.group(2))
    if ratio > 3 or ratio < 1 / 3:
        raise ImageInputError(
            f"Size {width}x{height} has an aspect ratio outside the supported range 1:3 to 3:1",
            param="size",
            code="invalid_aspect_ratio",
        )
    return width, height


def encode_image(image: PIL.Image.Image, output_format: str, compression: int = 100) -> str:
    buf = BytesIO()
    if output_format == "png":
        image.save(buf, "PNG")
    elif output_format == "jpeg":
        image.convert("RGB").save(buf, "JPEG", quality=max(1, compression))
    elif output_format == "webp":
        # compression=100 means "best quality", which for webp is the lossless mode
        image.save(buf, "WEBP", quality=compression, lossless=compression >= 100)
    else:
        raise ValueError(f"Unsupported output_format {output_format!r}")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def load_upload(data: bytes, filename: str | None = None) -> PIL.Image.Image:
    label = f" {filename!r}" if filename else ""
    if not data:
        raise ImageInputError(f"Uploaded file{label} is empty", param="image", code="invalid_image")
    try:
        image = PIL.Image.open(BytesIO(data))
        image = ImageOps.exif_transpose(image)
        return image.convert("RGB")
    except (UnidentifiedImageError, DecompressionBombError, OSError) as exc:
        raise ImageInputError(
            f"Uploaded file{label} is not a valid image ({type(exc).__name__})",
            param="image",
            code="invalid_image",
        ) from exc


def load_mask(data: bytes, filename: str | None = None) -> PIL.Image.Image:
    """Binary 'L' mask, 255 = repaint: transparent pixels (OpenAI), or white if no alpha."""
    label = f" {filename!r}" if filename else ""
    if not data:
        raise ImageInputError(f"Mask file{label} is empty", param="mask", code="invalid_mask")
    try:
        image = PIL.Image.open(BytesIO(data))
        image = ImageOps.exif_transpose(image)
        image.load()
    except (UnidentifiedImageError, DecompressionBombError, OSError) as exc:
        raise ImageInputError(
            f"Mask file{label} is not a valid image ({type(exc).__name__})",
            param="mask",
            code="invalid_mask",
        ) from exc
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        alpha = image.convert("RGBA").getchannel("A")
        return alpha.point(lambda a: 255 if a < 128 else 0)
    return image.convert("L").point(lambda v: 255 if v >= 128 else 0)


def composite_with_mask(
    generated: PIL.Image.Image, source: PIL.Image.Image, mask: PIL.Image.Image
) -> PIL.Image.Image:
    """Keep the source pixels wherever the mask is 0 (the model only owns the repainted area)."""
    size = generated.size
    src = source.convert("RGB").resize(size, PIL.Image.Resampling.LANCZOS)
    m = mask.convert("L").resize(size, PIL.Image.Resampling.BILINEAR)
    return PIL.Image.composite(generated.convert("RGB"), src, m)
