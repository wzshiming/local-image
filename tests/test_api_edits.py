import base64
from io import BytesIO

import PIL.Image
import pytest
from fastapi.testclient import TestClient

from flux_server.api import create_app
from flux_server.config import Settings
from tests.conftest import FakeEngine, png_bytes


def _file(name: str, data: bytes, mime: str = "image/png"):
    return (name, data, mime)


def _post(client, files, **fields):
    data = {k: str(v) for k, v in fields.items()}
    return client.post("/v1/images/edits", files=files, data=data)


def _decode(item: dict) -> PIL.Image.Image:
    return PIL.Image.open(BytesIO(base64.b64decode(item["b64_json"])))


def test_single_image(client, engine):
    r = _post(client, [("image", _file("ref.png", png_bytes(640, 480)))], prompt="make it blue")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["size"] == "640x480"
    assert _decode(body["data"][0]).size == (640, 480)
    job = engine.jobs[0]
    assert len(job.images) == 1
    assert isinstance(job.images[0], PIL.Image.Image)
    assert job.images[0].size == (640, 480)
    assert (job.width, job.height) == (None, None)
    assert job.prompt == "make it blue"


def test_image_array_key(client, engine):
    files = [
        ("image[]", _file("a.png", png_bytes(64, 64))),
        ("image[]", _file("b.png", png_bytes(32, 32))),
    ]
    r = _post(client, files, prompt="x")
    assert r.status_code == 200, r.text
    assert len(engine.jobs[0].images) == 2


def test_mixed_image_keys(client, engine):
    files = [
        ("image", _file("a.png", png_bytes())),
        ("image[]", _file("b.png", png_bytes())),
        ("image[]", _file("c.png", png_bytes())),
    ]
    r = _post(client, files, prompt="x")
    assert r.status_code == 200, r.text
    assert len(engine.jobs[0].images) == 3


def test_mixed_image_keys_keep_wire_order(client, engine):
    files = [
        ("image[]", _file("a.png", png_bytes(96, 64))),
        ("image", _file("b.png", png_bytes(64, 96))),
        ("image[]", _file("c.png", png_bytes(32, 32))),
    ]
    r = _post(client, files, prompt="x")
    assert r.status_code == 200, r.text
    assert [img.size for img in engine.jobs[0].images] == [(96, 64), (64, 96), (32, 32)]


def test_explicit_size_with_reference(client, engine):
    r = _post(client, [("image", _file("a.png", png_bytes(640, 480)))], prompt="x", size="512x512")
    assert r.status_code == 200
    assert r.json()["size"] == "512x512"
    assert (engine.jobs[0].width, engine.jobs[0].height) == (512, 512)


def _rgba_mask_bytes(w: int = 64, h: int = 64) -> bytes:
    """Left half transparent (= repaint per OpenAI), right half opaque."""
    mask = PIL.Image.new("RGBA", (w, h), (0, 0, 0, 255))
    for x in range(w // 2):
        for y in range(h):
            mask.putpixel((x, y), (0, 0, 0, 0))
    buf = BytesIO()
    mask.save(buf, "PNG")
    return buf.getvalue()


def test_mask_is_applied_to_first_image(client, engine):
    files = [
        ("image", _file("a.png", png_bytes(64, 64))),
        ("mask", _file("m.png", _rgba_mask_bytes(64, 64))),
    ]
    r = _post(client, files, prompt="x")
    assert r.status_code == 200, r.text
    job = engine.jobs[0]
    assert job.mask is not None and job.mask.mode == "L" and job.mask.size == (64, 64)
    assert job.mask.getpixel((0, 0)) == 255 and job.mask.getpixel((63, 0)) == 0
    assert job.strength is None and job.inpaint
    assert (job.width, job.height) == (None, None)


def test_strength_field(client, engine):
    r = _post(client, [("image", _file("a.png", png_bytes()))], prompt="x", strength="0.6")
    assert r.status_code == 200, r.text
    job = engine.jobs[0]
    assert job.strength == 0.6 and job.mask is None and job.inpaint


@pytest.mark.parametrize("value", ["0", "1.5", "-0.2", "abc"])
def test_strength_out_of_range(client, value):
    r = _post(client, [("image", _file("a.png", png_bytes()))], prompt="x", strength=value)
    assert r.status_code == 400
    assert r.json()["error"]["param"] == "strength"


def test_mask_requires_auto_size(client):
    files = [("image", _file("a.png", png_bytes())), ("mask", _file("m.png", png_bytes()))]
    r = _post(client, files, prompt="x", size="512x512")
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["param"] == "size" and err["code"] == "unsupported_parameter"
    assert _post(client, files, prompt="x", size="auto").status_code == 200


def test_strength_with_too_many_references(client):
    files = [("image[]", _file(f"{i}.png", png_bytes())) for i in range(3)]
    r = _post(client, files, prompt="x", strength="0.5")
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "too_many_images"
    assert _post(client, files[:2], prompt="x", strength="0.5").status_code == 200


def test_mask_must_be_an_image(client):
    files = [
        ("image", _file("a.png", png_bytes())),
        ("mask", _file("m.txt", b"nope", "text/plain")),
    ]
    r = _post(client, files, prompt="x")
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["param"] == "mask" and err["code"] == "invalid_mask"


def test_strength_ignored_by_generations(client, engine):
    r = client.post(
        "/v1/images/generations", json={"prompt": "x", "size": "64x64", "strength": 0.5}
    )
    assert r.status_code == 200
    assert engine.jobs[0].strength is None and not engine.jobs[0].inpaint


def test_no_files(client):
    # filename=None makes httpx send a plain multipart field
    r = client.post("/v1/images/edits", files={"prompt": (None, "x")})
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["param"] == "image"
    assert err["code"] == "missing_image"


def test_too_many_images():
    settings = Settings(_env_file=None, ui=False, max_ref_images=2)
    with TestClient(create_app(settings, engine=FakeEngine())) as c:
        files = [("image[]", _file(f"{i}.png", png_bytes())) for i in range(3)]
        r = _post(c, files, prompt="x")
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "too_many_images"


def test_upload_too_large():
    settings = Settings(_env_file=None, ui=False, max_upload_mb=1)
    with TestClient(create_app(settings, engine=FakeEngine())) as c:
        big = b"\0" * (settings.max_upload_bytes + 1)
        r = _post(c, [("image", _file("big.png", big))], prompt="x")
    assert r.status_code == 413
    err = r.json()["error"]
    assert err["code"] == "file_too_large" and err["param"] == "image"


def test_garbage_file(client):
    r = _post(client, [("image", _file("bad.png", b"definitely not an image"))], prompt="x")
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["code"] == "invalid_image"
    assert err["param"] == "image"
    assert "bad.png" in err["message"]


def test_string_fields_coerced(client, engine):
    r = _post(
        client, [("image", _file("a.png", png_bytes()))], prompt="x", n="2", seed="5", steps="3"
    )
    assert r.status_code == 200, r.text
    data = r.json()["data"]
    assert len(data) == 2
    assert [d["seed"] for d in data] == [5, 6]
    assert engine.jobs[0].steps == 3


def test_quality_string(client, engine):
    r = _post(client, [("image", _file("a.png", png_bytes()))], prompt="x", quality="high")
    assert r.status_code == 200
    assert engine.jobs[0].steps == 8
    assert r.json()["quality"] == "high"


def test_json_body_rejected(client):
    r = client.post("/v1/images/edits", json={"prompt": "x"})
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["type"] == "invalid_request_error"
    assert err["code"] == "invalid_content_type"


def test_missing_prompt(client):
    r = client.post("/v1/images/edits", files=[("image", _file("a.png", png_bytes()))])
    assert r.status_code == 400
    assert r.json()["error"]["param"] == "prompt"


def test_invalid_n_string(client):
    r = _post(client, [("image", _file("a.png", png_bytes()))], prompt="x", n="abc")
    assert r.status_code == 400
    assert r.json()["error"]["param"] == "n"


def test_response_format_url_rejected(client):
    r = _post(client, [("image", _file("a.png", png_bytes()))], prompt="x", response_format="url")
    assert r.status_code == 400
    assert r.json()["error"]["param"] == "response_format"
    assert r.json()["error"]["code"] == "unsupported_parameter"


def test_jpeg_upload_with_exif_is_transposed(client, engine):
    src = PIL.Image.new("RGB", (100, 50), "white")
    exif = PIL.Image.Exif()
    exif[274] = 6
    buf = BytesIO()
    src.save(buf, "JPEG", exif=exif.tobytes())
    r = _post(client, [("image", _file("r.jpg", buf.getvalue(), "image/jpeg"))], prompt="x")
    assert r.status_code == 200
    assert engine.jobs[0].images[0].size == (50, 100)
    assert r.json()["size"] == "50x100"
