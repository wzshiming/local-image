import base64
from io import BytesIO

import PIL.Image
import pytest

from flux_server.imaging import (
    ImageInputError,
    composite_with_mask,
    encode_image,
    load_mask,
    load_upload,
    parse_size,
)

MAX = 2048 * 2048


@pytest.mark.parametrize("size", [None, "", "auto", "AUTO", " Auto "])
def test_parse_size_auto(size):
    assert parse_size(size, MAX) == (None, None)


@pytest.mark.parametrize(
    ("size", "expected"),
    [
        ("512x512", (512, 512)),
        ("1010x600", (1008, 608)),
        ("1024X768", (1024, 768)),
        (" 1024 x 768 ", (1024, 768)),
        ("1024×768", (1024, 768)),
        ("48x48", (64, 64)),
    ],
)
def test_parse_size_valid(size, expected):
    assert parse_size(size, MAX) == expected


@pytest.mark.parametrize(
    "size", ["abc", "10x", "0x512", "512x-5", "512", "x", "1.5x2", "1" + "0" * 400 + "x1024"]
)
def test_parse_size_malformed(size):
    with pytest.raises(ImageInputError) as info:
        parse_size(size, MAX)
    assert info.value.code == "invalid_size"
    assert info.value.param == "size"


def test_parse_size_too_large():
    with pytest.raises(ImageInputError) as info:
        parse_size("2048x2064", MAX)
    assert info.value.code == "size_too_large"
    assert parse_size("2048x2048", MAX) == (2048, 2048)


def test_parse_size_aspect_ratio():
    with pytest.raises(ImageInputError) as info:
        parse_size("2048x256", MAX)
    assert info.value.code == "invalid_aspect_ratio"
    with pytest.raises(ImageInputError):
        parse_size("256x2048", MAX)
    assert parse_size("1536x512", MAX) == (1536, 512)
    assert parse_size("512x1536", MAX) == (512, 1536)


def test_parse_size_ratio_checked_before_rounding():
    assert parse_size("300x100", MAX) == (304, 96)
    assert parse_size("100x300", MAX) == (96, 304)
    with pytest.raises(ImageInputError) as info:
        parse_size("301x100", MAX)
    assert info.value.code == "invalid_aspect_ratio"


def _decode(b64: str) -> PIL.Image.Image:
    return PIL.Image.open(BytesIO(base64.b64decode(b64)))


@pytest.mark.parametrize(
    ("fmt", "pil_format"), [("png", "PNG"), ("jpeg", "JPEG"), ("webp", "WEBP")]
)
def test_encode_image_round_trip(fmt, pil_format):
    src = PIL.Image.new("RGB", (96, 64), "blue")
    out = _decode(encode_image(src, fmt))
    assert out.format == pil_format
    assert out.size == (96, 64)
    assert out.convert("RGB").mode == "RGB"


def test_encode_image_jpeg_compression_reduces_size():
    src = PIL.Image.radial_gradient("L").convert("RGB").resize((256, 256))
    low = encode_image(src, "jpeg", 10)
    high = encode_image(src, "jpeg", 100)
    assert len(low) < len(high)


def test_encode_image_jpeg_zero_compression_is_clamped():
    src = PIL.Image.new("RGB", (32, 32), "green")
    assert _decode(encode_image(src, "jpeg", 0)).format == "JPEG"


def test_encode_image_webp_100_is_lossless():
    src = PIL.Image.radial_gradient("L").convert("RGB").resize((64, 64))
    out = _decode(encode_image(src, "webp", 100))
    assert list(out.convert("RGB").getdata()) == list(src.getdata())


def test_encode_image_rejects_unknown_format():
    with pytest.raises(ValueError):
        encode_image(PIL.Image.new("RGB", (8, 8)), "gif")


def _bytes(image: PIL.Image.Image, fmt: str, **kwargs) -> bytes:
    buf = BytesIO()
    image.save(buf, fmt, **kwargs)
    return buf.getvalue()


def test_load_upload_png_rgb():
    img = load_upload(_bytes(PIL.Image.new("RGB", (40, 30), "red"), "PNG"), "a.png")
    assert img.mode == "RGB"
    assert img.size == (40, 30)


def test_load_upload_rgba_converted():
    img = load_upload(_bytes(PIL.Image.new("RGBA", (20, 10), (1, 2, 3, 128)), "PNG"))
    assert img.mode == "RGB"
    assert img.size == (20, 10)


def test_load_upload_exif_orientation_transposed():
    src = PIL.Image.new("RGB", (100, 50), "white")
    exif = PIL.Image.Exif()
    exif[274] = 6
    img = load_upload(_bytes(src, "JPEG", exif=exif.tobytes()), "rot.jpg")
    assert img.size == (50, 100)


@pytest.mark.parametrize("data", [b"", b"not an image"])
def test_load_upload_invalid(data):
    with pytest.raises(ImageInputError) as info:
        load_upload(data, "bad.bin")
    assert info.value.code == "invalid_image"
    assert info.value.param == "image"
    assert "bad.bin" in str(info.value)


def test_load_upload_decompression_bomb(monkeypatch):
    monkeypatch.setattr(PIL.Image, "MAX_IMAGE_PIXELS", 10)
    with pytest.raises(ImageInputError) as info:
        load_upload(_bytes(PIL.Image.new("RGB", (64, 64)), "PNG"), "bomb.png")
    assert info.value.code == "invalid_image"


def test_load_mask_alpha_transparent_means_repaint():
    rgba = PIL.Image.new("RGBA", (4, 2), (255, 255, 255, 255))
    rgba.putpixel((0, 0), (255, 255, 255, 0))
    rgba.putpixel((1, 0), (0, 0, 0, 40))
    mask = load_mask(_bytes(rgba, "PNG"), "m.png")
    assert mask.mode == "L" and mask.size == (4, 2)
    assert mask.getpixel((0, 0)) == 255 and mask.getpixel((1, 0)) == 255
    assert mask.getpixel((2, 0)) == 0 and mask.getpixel((3, 1)) == 0


def test_load_mask_without_alpha_white_means_repaint():
    rgb = PIL.Image.new("RGB", (2, 1), "black")
    rgb.putpixel((1, 0), (255, 255, 255))
    mask = load_mask(_bytes(rgb, "JPEG"))
    assert mask.mode == "L"
    assert mask.getpixel((0, 0)) == 0 and mask.getpixel((1, 0)) == 255


@pytest.mark.parametrize("data", [b"", b"not an image"])
def test_load_mask_invalid(data):
    with pytest.raises(ImageInputError) as info:
        load_mask(data, "m.bin")
    assert info.value.code == "invalid_mask" and info.value.param == "mask"


def test_composite_with_mask_keeps_source_outside_mask():
    generated = PIL.Image.new("RGB", (8, 4), (255, 0, 0))
    source = PIL.Image.new("RGB", (16, 8), (0, 0, 255))  # different size: resized to output
    mask = PIL.Image.new("L", (16, 8), 0)
    for x in range(8):
        for y in range(8):
            mask.putpixel((x, y), 255)
    out = composite_with_mask(generated, source, mask)
    assert out.size == (8, 4)
    assert out.getpixel((0, 0)) == (255, 0, 0)
    assert out.getpixel((7, 3)) == (0, 0, 255)
