import asyncio
import base64
import json
import os
import shlex
import threading
from pathlib import Path
from types import SimpleNamespace

import gradio as gr
import gradio.processing_utils
import httpx
import openai
import PIL.Image
import PIL.ImageDraw
import pytest
from fastapi.testclient import TestClient
from gradio.components.gallery import GalleryData

from flux_server.api import create_app
from flux_server.config import Settings
from flux_server.imaging import load_mask
from flux_server.ui import (
    EDIT_PLACEHOLDER,
    GenParams,
    build_generate_kwargs,
    build_ui,
    cancel_request,
    curl_for_edit,
    curl_for_generate,
    data_url,
    decode_data_url,
    decode_images,
    error_code,
    format_error,
    gallery_paths,
    gallery_value,
    iter_edit,
    iter_generate,
    mask_from_editor,
    normalize_api_base,
    progress_text,
    request_preview,
    resolve_size,
    run_edit,
    run_generate,
    select_gallery_image,
    status_line,
    write_temp_mask,
)
from tests.conftest import FakeEngine, png_bytes

API_BASE = "http://testserver/v1"
MODEL = "flux.2-klein-4b"


def params(**overrides) -> GenParams:
    base = dict(
        prompt="x", size="256x256", quality=None, steps=None, seed=None, n=1, output_format="png"
    )
    base.update(overrides)
    return GenParams(**base)


class TestBuildGenerateKwargs:
    def test_minimal_omits_optional_keys(self):
        kw = build_generate_kwargs(MODEL, params(size="auto", quality="auto"))
        assert kw == {"model": MODEL, "prompt": "x", "n": 1, "output_format": "png"}

    def test_none_size_and_quality_omitted(self):
        kw = build_generate_kwargs(MODEL, params(size=None, quality=None))
        assert "size" not in kw and "quality" not in kw and "extra_body" not in kw

    def test_full(self):
        kw = build_generate_kwargs(
            MODEL,
            params(size="512x512", quality="high", steps=8, seed=3, n=2, output_format="webp"),
        )
        assert kw["size"] == "512x512"
        assert kw["quality"] == "high"
        assert kw["n"] == 2
        assert kw["output_format"] == "webp"
        assert kw["extra_body"] == {"seed": 3, "steps": 8}

    def test_seed_zero_kept_negative_dropped(self):
        assert build_generate_kwargs(MODEL, params(seed=0))["extra_body"] == {"seed": 0}
        assert "extra_body" not in build_generate_kwargs(MODEL, params(seed=-1))

    def test_steps_zero_dropped(self):
        assert "extra_body" not in build_generate_kwargs(MODEL, params(steps=0))

    def test_model_override(self):
        assert build_generate_kwargs(MODEL, params(model="gpt-image-1"))["model"] == "gpt-image-1"
        assert build_generate_kwargs(MODEL, params(model=None))["model"] == MODEL

    def test_compression_only_for_lossy_formats(self):
        assert "output_compression" not in build_generate_kwargs(MODEL, params(output_format="png"))
        kw = build_generate_kwargs(MODEL, params(output_format="jpeg", output_compression=80))
        assert kw["output_compression"] == 80

    def test_strength_goes_to_extra_body(self):
        assert build_generate_kwargs(MODEL, params(strength=0.5))["extra_body"] == {"strength": 0.5}
        assert "extra_body" not in build_generate_kwargs(MODEL, params(strength=None))
        assert "extra_body" not in build_generate_kwargs(MODEL, params(strength=0))


class TestResolveSize:
    def test_custom_uses_width_height(self):
        assert resolve_size("custom", 832, 1216) == "832x1216"

    def test_custom_without_dimensions(self):
        assert resolve_size("custom", None, 512) is None

    def test_passthrough(self):
        assert resolve_size("1024x768", 256, 256) == "1024x768"
        assert resolve_size("auto", 256, 256) == "auto"
        assert resolve_size("", 256, 256) is None
        assert resolve_size(None, 256, 256) is None


class TestNormalizeApiBase:
    def test_strips_slash_and_whitespace(self):
        assert normalize_api_base(" http://x/v1/ ", "d") == "http://x/v1"

    def test_empty_falls_back(self):
        assert normalize_api_base("", "d") == "d"
        assert normalize_api_base(None, "d") == "d"


class TestRequestPreview:
    def test_generate(self):
        meta, curl = request_preview(API_BASE, MODEL, params(seed=7, steps=2, size="512x512"))
        assert meta["endpoint"] == f"{API_BASE}/images/generations"
        assert meta["request"]["seed"] == 7 and meta["request"]["steps"] == 2
        assert meta["request"]["size"] == "512x512" and "extra_body" not in meta["request"]
        assert curl == curl_for_generate(
            API_BASE, build_generate_kwargs(MODEL, params(seed=7, steps=2, size="512x512"))
        )

    def test_edit_without_files_uses_placeholder(self):
        meta, curl = request_preview(API_BASE, MODEL, params(), filenames=[])
        assert meta["endpoint"] == f"{API_BASE}/images/edits"
        assert meta["request"]["image[]"] == [EDIT_PLACEHOLDER]
        assert f"'image[]=@{EDIT_PLACEHOLDER}'" in curl

    def test_edit_with_files(self):
        meta, curl = request_preview(API_BASE, MODEL, params(), filenames=["a.png", "b.png"])
        assert meta["request"]["image[]"] == ["a.png", "b.png"]
        assert "'image[]=@a.png'" in curl and "'image[]=@b.png'" in curl
        assert "mask" not in meta["request"] and "mask=@" not in curl

    def test_edit_with_mask_and_strength(self):
        meta, curl = request_preview(
            API_BASE, MODEL, params(strength=0.6), filenames=["a.png"], mask="m.png"
        )
        assert meta["request"]["mask"] == "m.png" and meta["request"]["strength"] == 0.6
        assert "-F mask=@m.png" in curl and "strength=0.6" in curl


def test_build_ui_constructs_without_server():
    demo = build_ui(Settings(_env_file=None))
    blocks = list(demo.blocks.values())
    labels = {getattr(b, "label", None) for b in blocks}
    assert {"API base", "model", "等价 curl", "结果"} <= labels
    assert any(str(label).startswith("请求参数") for label in labels)
    assert any(str(label).startswith("响应元数据") for label in labels)
    buttons = [b.value for b in blocks if isinstance(b, gr.Button)]
    assert buttons.count("取消") == 2
    assert "发送到图生图" in buttons and "用选中结果继续编辑" in buttons
    assert "strength" in labels
    assert any(isinstance(b, gr.Tabs) for b in blocks)
    assert {t.id for t in blocks if isinstance(t, gr.Tab)} == {"generate", "edit", "status"}
    assert any(isinstance(b, gr.Dataset) for b in blocks)  # gr.Examples
    galleries = [b for b in blocks if isinstance(b, gr.Gallery)]
    assert [g.interactive for g in galleries].count(True) == 1  # only the refs uploader
    assert all(type(g) is gr.Gallery for g in galleries)  # no Component subclass (no .pyi stub)
    editors = [b for b in blocks if isinstance(b, gr.ImageEditor)]
    assert len(editors) == 1 and editors[0].sources == () and editors[0].type == "pil"
    fns = list(demo.fns.values())
    api_names = {fn.api_name for fn in fns}
    assert {"generate", "edit", "status"} <= api_names
    runs = [fn for fn in fns if fn.api_name in {"generate", "edit"}]
    assert len(runs) == 2 and all(fn.postprocess is False for fn in runs)
    others = [fn for fn in fns if fn not in runs and fn.api_name != "load_example"]  # gr.Examples
    assert all(fn.postprocess is True for fn in others)
    assert sum(fn.preprocess is False for fn in fns) == 2  # send / reuse read raw gallery dicts
    assert all(fn.preprocess is True for fn in runs)


RECT = (10, 20, 40, 50)  # x0, y0, x1, y1 (x1/y1 inclusive for ImageDraw)


def painted_layer(size=(64, 64), box=RECT, color=(255, 60, 60, 153)) -> PIL.Image.Image:
    layer = PIL.Image.new("RGBA", size, (0, 0, 0, 0))
    PIL.ImageDraw.Draw(layer).rectangle(box, fill=color)
    return layer


def assert_rect_mask(mask: PIL.Image.Image, size=(64, 64), box=RECT) -> None:
    assert mask.mode == "L" and mask.size == size
    x0, y0, x1, y1 = box
    assert mask.getpixel((x0, y0)) == 255 and mask.getpixel((x1, y1)) == 255
    assert mask.getpixel((x0 - 1, y0)) == 0 and mask.getpixel((x1 + 1, y1 + 1)) == 0
    assert mask.getbbox() == (x0, y0, x1 + 1, y1 + 1)


class TestMaskFromEditor:
    bg = PIL.Image.new("RGBA", (64, 64), "blue")

    def test_none_and_empty(self):
        assert mask_from_editor(None) is None
        assert mask_from_editor({"background": self.bg, "layers": [], "composite": None}) is None
        assert mask_from_editor({"background": None, "layers": [None]}) is None

    def test_transparent_layer_is_none(self):
        empty = PIL.Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        assert mask_from_editor({"background": self.bg, "layers": [empty]}) is None

    def test_painted_rectangle(self):
        mask = mask_from_editor({"background": self.bg, "layers": [painted_layer()]})
        assert mask is not None
        assert_rect_mask(mask)
        assert mask.getextrema() == (0, 255)

    def test_layer_resized_to_background(self):
        layer = painted_layer(size=(32, 32), box=(5, 10, 19, 24))
        mask = mask_from_editor({"background": self.bg, "layers": [layer]})
        assert mask is not None and mask.size == (64, 64)
        assert mask.getpixel((10, 20)) == 255 and mask.getpixel((39, 49)) == 255
        assert mask.getpixel((9, 20)) == 0 and mask.getpixel((40, 50)) == 0

    def test_no_background_uses_layer_size(self):
        mask = mask_from_editor({"background": None, "layers": [painted_layer((48, 40))]})
        assert mask is not None and mask.size == (48, 40)

    def test_two_layers_union(self):
        a = painted_layer(box=(0, 0, 9, 9))
        b = painted_layer(box=(50, 50, 63, 63))
        mask = mask_from_editor({"background": self.bg, "layers": [a, b]})
        assert mask is not None
        assert mask.getpixel((5, 5)) == 255 and mask.getpixel((60, 60)) == 255
        assert mask.getpixel((30, 30)) == 0


def test_write_temp_mask_round_trips_through_server(sdk, tmp_path: Path):
    client, engine = sdk
    mask = mask_from_editor({"background": None, "layers": [painted_layer()]})
    assert mask is not None
    tmp = write_temp_mask(mask)
    try:
        assert tmp.endswith(".png")
        assert_rect_mask(load_mask(Path(tmp).read_bytes()))

        ref = tmp_path / "ref.png"
        ref.write_bytes(png_bytes(64, 64, "red"))
        result = run_edit(client, MODEL, API_BASE, [str(ref)], params(size=None), mask_path=tmp)
        assert len(result.images) == 1 and f"mask=@{tmp}" in result.curl
        job = engine.jobs[-1]
        assert job.mask is not None
        assert_rect_mask(job.mask)
    finally:
        os.unlink(tmp)


def _edit_handlers(demo):
    fns = {getattr(fn.fn, "__name__", ""): fn.fn for fn in demo.fns.values()}
    return fns["on_edit"], fns["on_edit_preview"], fns["on_refs_change"]


def _handlers(demo):
    return {getattr(fn.fn, "__name__", ""): fn.fn for fn in demo.fns.values()}


def _png_data_url(w: int, h: int, color="red") -> str:
    return data_url(base64.b64encode(png_bytes(w, h, color)).decode(), "png")


def test_data_url_round_trip():
    b64 = base64.b64encode(png_bytes(8, 4)).decode()
    assert data_url(b64, "png").startswith("data:image/png;base64,")
    assert data_url(b64, "jpeg").startswith("data:image/jpeg;base64,")
    assert data_url(b64, "webp").startswith("data:image/webp;base64,")
    assert data_url(b64, "bmp").startswith("data:image/png;base64,")  # unknown -> png
    assert decode_data_url(data_url(b64, "png")).size == (8, 4)
    with pytest.raises(ValueError):
        decode_data_url("/tmp/some/file.png")


class TestGalleryValue:
    def test_pre_serialized_data_urls(self):
        u1, u2 = _png_data_url(8, 4), _png_data_url(4, 8, "blue")
        value = gallery_value([u1, u2], ["seed=7", "seed=?"])
        assert isinstance(value, list) and len(value) == 2
        assert all(isinstance(item, dict) for item in value)
        first, second = value
        assert first["image"]["path"] is None and first["image"]["url"] == u1
        assert first["image"]["orig_name"] == "seed-7.png"
        assert first["image"]["mime_type"] == "image/png"
        assert first["image"]["meta"] == {"_type": "gradio.FileData"}
        assert first["caption"] == "seed=7"
        assert second["image"]["path"] is None and second["image"]["url"] == u2
        assert second["image"]["orig_name"] == "image-2.png"
        assert second["caption"] == "seed=?"

    def test_survives_gradio_output_cache_untouched(self):
        value = gallery_value([_png_data_url(8, 4)], ["seed=7"])
        moved = asyncio.run(
            gradio.processing_utils.async_move_files_to_cache(value, gr.Gallery(), postprocess=True)
        )
        assert moved == value
        assert "file=" not in json.dumps(moved) and "gradio_api" not in json.dumps(moved)

    def test_validates_as_gallery_data(self):
        value = gallery_value([_png_data_url(2, 2)], ["seed=1"])
        model = gr.Gallery().data_model
        assert model is GalleryData
        assert model(root=value).model_dump() == value

    def test_extension_follows_mime(self):
        b64 = base64.b64encode(png_bytes(2, 2)).decode()
        value = gallery_value([data_url(b64, "webp")], ["seed=3"])
        assert value[0]["image"]["orig_name"] == "seed-3.webp"
        assert value[0]["image"]["mime_type"] == "image/webp"

    def test_empty(self):
        assert gallery_value([], []) == []


def test_on_generate_yields_data_urls_and_on_send_decodes_them(monkeypatch):
    engine = FakeEngine()
    app = create_app(Settings(_env_file=None, ui=False), engine=engine)
    fns = _handlers(build_ui(Settings(_env_file=None)))
    on_generate, on_send = fns["on_generate"], fns["on_send"]
    import flux_server.ui as ui_module

    with TestClient(app) as tc:
        monkeypatch.setattr(
            ui_module,
            "make_client",
            lambda base: openai.OpenAI(
                base_url=API_BASE, api_key="local", max_retries=0, http_client=tc
            ),
        )
        controls = ("256x256", None, None, None, 2, 7, 2, "png", 100)
        outs = list(on_generate("rid", API_BASE, MODEL, "p", *controls))

    assert outs[0][0] == gr.skip() and len(outs[0]) == 3
    items, meta, status = outs[-1]
    assert status.startswith("✅ 完成") and meta["done"] == 2
    assert isinstance(items, list) and all(isinstance(item, dict) for item in items)
    assert [item["caption"] for item in items] == ["seed=7", "seed=8"]
    for item in items:
        url = item["image"]["url"]
        assert url.startswith("data:image/") and item["image"]["path"] is None
        assert "gradio_api" not in url and "file=" not in url and "\\" not in url
    assert len(outs[1][0]) == 1  # first image shown before the second is requested

    imgs, tabs = on_send(items, 1)
    assert len(imgs) == 1 and isinstance(imgs[0], PIL.Image.Image)
    assert imgs[0].size == (256, 256)
    assert isinstance(tabs, gr.Tabs)
    assert on_send([], 0)[0] == []


def test_on_edit_uses_painted_mask_and_removes_temp_file(monkeypatch, tmp_path: Path):
    engine = FakeEngine()
    app = create_app(Settings(_env_file=None, ui=False), engine=engine)
    demo = build_ui(Settings(_env_file=None))
    on_edit, on_edit_preview, _ = _edit_handlers(demo)
    written: list[str] = []
    import flux_server.ui as ui_module

    real_write = ui_module.write_temp_mask

    def spy_write(mask):
        written.append(real_write(mask))
        return written[-1]

    monkeypatch.setattr(ui_module, "write_temp_mask", spy_write)
    with TestClient(app) as tc:
        monkeypatch.setattr(
            ui_module,
            "make_client",
            lambda base: openai.OpenAI(
                base_url=API_BASE, api_key="local", max_retries=0, http_client=tc
            ),
        )
        ref = tmp_path / "ref.png"
        ref.write_bytes(png_bytes(64, 64, "red"))
        uploaded = tmp_path / "upload.png"
        uploaded.write_bytes(png_bytes(64, 64, "white"))
        refs = [(str(ref), None)]
        editor = {"background": PIL.Image.open(ref), "layers": [painted_layer()], "composite": None}
        controls = (None, None, None, None, 2, 7, 1, "png", 100)

        meta, curl = on_edit_preview(
            API_BASE, MODEL, refs, editor, str(uploaded), 0, "p", *controls
        )
        assert meta["request"]["mask"] == "<painted>" and "mask=@<painted>" in curl

        outs = list(on_edit("rid", API_BASE, MODEL, refs, editor, str(uploaded), 0, "p", *controls))
        assert len(written) == 1 and not os.path.exists(written[0])
        assert outs[-1][2].startswith("✅ 完成")
        items = outs[-1][0]
        assert isinstance(items, list) and [type(item) for item in items] == [dict]
        assert items[0]["image"]["url"].startswith("data:image/png;base64,")
        assert items[0]["image"]["path"] is None and items[0]["caption"] == "seed=7"
        assert_rect_mask(engine.jobs[-1].mask)  # painted wins over the uploaded file

        # no painting -> the uploaded file is used
        editor = {"background": PIL.Image.open(ref), "layers": [], "composite": None}
        meta, _ = on_edit_preview(API_BASE, MODEL, refs, editor, str(uploaded), 0, "p", *controls)
        assert meta["request"]["mask"] == str(uploaded)
        list(on_edit("rid2", API_BASE, MODEL, refs, editor, str(uploaded), 0, "p", *controls))
        assert len(written) == 1 and engine.jobs[-1].mask.getextrema() == (255, 255)

        # neither -> no mask
        list(on_edit("rid3", API_BASE, MODEL, refs, None, None, 0, "p", *controls))
        assert engine.jobs[-1].mask is None

        # server error mid-run: gr.Error is raised and the temp file is still removed
        editor = {"background": PIL.Image.open(ref), "layers": [painted_layer()], "composite": None}
        bad = ("512x512", None, None, None, 2, 7, 1, "png", 100)  # size must be auto with mask
        with pytest.raises(gr.Error):
            list(on_edit("rid4", API_BASE, MODEL, refs, editor, None, 0, "p", *bad))
        assert len(written) == 2 and not os.path.exists(written[1])


def test_on_refs_change_only_reloads_when_first_ref_changes():
    _, _, on_refs_change = _edit_handlers(build_ui(Settings(_env_file=None)))
    editor, first = on_refs_change([("/a.png", None)], None)
    assert (editor, first) == ("/a.png", "/a.png")
    editor, first = on_refs_change([("/a.png", None), ("/b.png", None)], "/a.png")
    assert editor == gr.skip() and first == "/a.png"
    assert on_refs_change([], "/a.png") == (None, None)
    assert on_refs_change(None, None)[1] is None


class TestGalleryHelpers:
    def test_gallery_paths(self):
        assert gallery_paths(None) == []
        assert gallery_paths([]) == []
        assert gallery_paths([("/a.png", None), ("/b.png", "seed=1")]) == ["/a.png", "/b.png"]
        assert gallery_paths(["/a.png", {"image": {"path": "/b.png"}}, {"path": "/c.png"}]) == [
            "/a.png",
            "/b.png",
            "/c.png",
        ]

    def test_select_raw_gallery_dicts(self):
        items = gallery_value([_png_data_url(8, 4), _png_data_url(4, 8)], ["seed=1", "seed=2"])
        assert [img.size for img in select_gallery_image(items, 1)] == [(4, 8)]
        assert [img.size for img in select_gallery_image(items, 0)] == [(8, 4)]

    def test_select_tuple_items(self):
        items = [(_png_data_url(8, 4), "seed=1"), (_png_data_url(4, 8), "seed=2")]
        assert [img.size for img in select_gallery_image(items, 1)] == [(4, 8)]
        assert [img.size for img in select_gallery_image(items, 0)] == [(8, 4)]

    def test_select_str_items_and_default_index(self):
        items = [_png_data_url(8, 4), _png_data_url(4, 8)]
        assert [img.size for img in select_gallery_image(items, None)] == [(8, 4)]

    def test_select_clamps(self):
        items = gallery_value([_png_data_url(8, 4), _png_data_url(4, 8)], ["seed=1", "seed=2"])
        assert select_gallery_image(items, 7)[0].size == (4, 8)
        assert select_gallery_image(items, -3)[0].size == (8, 4)
        assert select_gallery_image(items, "x")[0].size == (8, 4)

    def test_select_empty(self):
        assert select_gallery_image([], 0) == []
        assert select_gallery_image(None, 2) == []

    def test_select_item_without_data_url_raises(self):
        items = [{"image": {"path": "/tmp/x.png", "url": "http://h/file=/tmp/x.png"}}]
        with pytest.raises(gr.Error):
            select_gallery_image(items, 0)
        with pytest.raises(gr.Error):
            select_gallery_image([("/tmp/x.png", None)], 0)


class TestStatusLine:
    def test_formats_health(self):
        health = {"status": "ok", "ready": True, "in_flight": 0, **FakeEngine().info()}
        assert status_line(health) == (
            "**flux.2-klein-4b** · cpu · float32 · offload none · 默认 4 步 · 进行中 0"
        )

    def test_in_flight_and_missing_fields(self):
        assert status_line({"model_id": "m", "in_flight": 2}) == "**m** · 进行中 2"
        assert status_line({}) == "**?** · 进行中 0"


class TestCurl:
    def test_generate(self):
        kw = build_generate_kwargs(MODEL, params(prompt="a 'quoted' cat", seed=7, steps=2))
        cmd = curl_for_generate(API_BASE, kw)
        assert f"{API_BASE}/images/generations" in cmd
        assert "-d" in cmd
        assert "extra_body" not in cmd
        body = shlex.split(cmd)[-1]
        assert '"seed": 7' in body and '"steps": 2' in body
        assert "a 'quoted' cat" in body

    def test_edit(self):
        kw = build_generate_kwargs(MODEL, params(prompt="it's snowy", size=None, seed=1, steps=3))
        cmd = curl_for_edit(API_BASE, kw, ["a.png", "b.png"])
        assert f"{API_BASE}/images/edits" in cmd
        assert "'image[]=@a.png'" in cmd and "'image[]=@b.png'" in cmd
        tokens = shlex.split(cmd)
        fields = [tokens[i + 1] for i, t in enumerate(tokens) if t == "-F"]
        assert "prompt=it's snowy" in fields
        assert "seed=1" in fields and "steps=3" in fields
        assert f"model={MODEL}" in fields
        assert not any(f.startswith("extra_body") for f in fields)


def test_decode_images():
    items = [
        SimpleNamespace(b64_json=base64.b64encode(png_bytes(8, 4)).decode(), seed=7),
        SimpleNamespace(b64_json=base64.b64encode(png_bytes(4, 8)).decode()),
    ]
    out = decode_images(items)
    assert [img.size for img, _ in out] == [(8, 4), (4, 8)]
    assert [cap for _, cap in out] == ["seed=7", "seed=?"]


class TestFormatError:
    def test_status_error_nested_body(self):
        exc = openai.APIStatusError(
            "m",
            response=httpx.Response(400, request=httpx.Request("POST", "http://x")),
            body={"error": {"message": "boom"}},
        )
        assert format_error(exc) == "HTTP 400: boom"

    def test_status_error_flat_body(self):
        exc = openai.APIStatusError(
            "m",
            response=httpx.Response(404, request=httpx.Request("POST", "http://x")),
            body={"message": "nope", "type": "invalid_request_error"},
        )
        assert format_error(exc) == "HTTP 404: nope"

    def test_status_error_no_body(self):
        exc = openai.APIStatusError(
            "raw text",
            response=httpx.Response(500, request=httpx.Request("POST", "http://x")),
            body=None,
        )
        assert format_error(exc) == "HTTP 500: raw text"

    def test_connection_error(self):
        exc = openai.APIConnectionError(request=httpx.Request("POST", "http://x"))
        assert format_error(exc).startswith("无法连接 API: ")

    def test_generic(self):
        assert format_error(ValueError("bad")) == "ValueError: bad"


@pytest.fixture
def sdk():
    engine = FakeEngine()
    app = create_app(Settings(_env_file=None, ui=False), engine=engine)
    with TestClient(app) as tc:
        client = openai.OpenAI(base_url=API_BASE, api_key="local", max_retries=0, http_client=tc)
        yield client, engine


def test_run_generate_end_to_end(sdk):
    client, engine = sdk
    p = params(prompt="x", size="256x256", steps=3, seed=5, n=2)
    result = run_generate(client, MODEL, API_BASE, p, request_id="rid-1")

    assert [img.size for img, _ in result.images] == [(256, 256), (256, 256)]
    assert [cap for _, cap in result.images] == ["seed=5", "seed=6"]
    assert len(result.urls) == 2
    for url, (img, _) in zip(result.urls, result.images, strict=True):
        assert url.startswith("data:image/png;base64,")
        assert decode_data_url(url).size == img.size
    assert result.meta["seeds"] == [5, 6]
    assert result.meta["size"] == "256x256"
    assert result.meta["steps_requested"] == 3
    assert result.meta["n"] == 2
    assert result.meta["refs"] == 0
    assert result.meta["output_format"] == "png"
    assert result.meta["request_id"] == "rid-1"
    assert isinstance(result.meta["elapsed_server_seconds"], float)
    assert result.meta["elapsed_server_seconds"] == pytest.approx(0.25)
    assert isinstance(result.meta["elapsed_client_seconds"], float)
    assert isinstance(result.meta["created"], int)
    assert "/images/generations" in result.curl

    job = engine.jobs[-1]
    assert (job.steps, job.seed, job.n, job.width, job.height) == (3, 5, 2, 256, 256)
    assert job.images is None


def test_run_edit_end_to_end(sdk, tmp_path: Path):
    client, engine = sdk
    paths = []
    for name, color in (("a.png", "red"), ("b.png", "blue")):
        path = tmp_path / name
        path.write_bytes(png_bytes(96, 64, color))
        paths.append(str(path))

    result = run_edit(client, MODEL, API_BASE, paths, params(size=None, seed=11))

    assert len(result.images) == 1
    assert result.images[0][0].size == (96, 64)
    assert result.images[0][1] == "seed=11"
    assert result.meta["refs"] == 2
    assert result.meta["size"] == "96x64"
    assert len(result.meta["request_id"]) == 32  # server-generated when the UI sends none
    assert "image[]=@" in result.curl and "/images/edits" in result.curl

    job = engine.jobs[-1]
    assert job.images is not None and len(job.images) == 2
    assert job.images[0].size == (96, 64)
    assert job.seed == 11 and job.width is None and job.height is None
    assert job.mask is None and job.strength is None


def test_run_edit_with_mask_and_strength(sdk, tmp_path: Path):
    client, engine = sdk
    src = tmp_path / "src.png"
    src.write_bytes(png_bytes(64, 64, "red"))
    mask = tmp_path / "mask.png"
    mask.write_bytes(png_bytes(64, 64, "white"))  # no alpha: white = repaint

    result = run_edit(
        client, MODEL, API_BASE, [str(src)], params(size=None, strength=0.4), mask_path=str(mask)
    )

    assert f"-F {shlex.quote(f'mask=@{mask}')}" in result.curl and "strength=0.4" in result.curl
    job = engine.jobs[-1]
    assert job.strength == 0.4
    assert job.mask is not None and job.mask.getextrema() == (255, 255)


def test_run_generate_api_error_surfaces_as_status_error(sdk):
    client, _ = sdk
    with pytest.raises(openai.APIStatusError) as info:
        run_generate(client, MODEL, API_BASE, params(n=99))
    assert "HTTP 400" in format_error(info.value)
    assert "n must be between" in format_error(info.value)
    assert error_code(info.value) == "invalid_value"


def test_iter_generate_streams_one_image_per_request(sdk):
    client, engine = sdk
    results = list(iter_generate(client, MODEL, API_BASE, params(seed=5, n=3), request_id="s-1"))

    assert [len(r.images) for r in results] == [1, 2, 3]
    assert [len(r.urls) for r in results] == [1, 2, 3]
    for url, (img, _) in zip(results[-1].urls, results[-1].images, strict=True):
        assert url.startswith("data:image/png;base64,")
        assert decode_data_url(url).size == img.size
    assert [cap for _, cap in results[-1].images] == ["seed=5", "seed=6", "seed=7"]
    last = results[-1].meta
    assert last["seeds"] == [5, 6, 7] and (last["n"], last["done"]) == (3, 3)
    assert last["elapsed_server_seconds"] == pytest.approx(0.75)
    assert last["request_id"] == "s-1" and "cancelled" not in last
    assert results[0].meta["done"] == 1
    # Same images as one n=3 request: the server would also use seeds 5, 6, 7.
    assert [(j.n, j.seed) for j in engine.jobs] == [(1, 5), (1, 6), (1, 7)]
    assert '"n": 3' in results[-1].curl


def test_iter_generate_chains_server_chosen_seed(sdk):
    client, engine = sdk
    results = list(iter_generate(client, MODEL, API_BASE, params(seed=None, n=2)))
    assert results[-1].meta["seeds"] == [12345, 12346]  # FakeEngine picks 12345 for seed=None
    assert [j.seed for j in engine.jobs] == [None, 12346]


def test_iter_generate_seed_wraps_like_the_server(sdk):
    client, engine = sdk
    list(iter_generate(client, MODEL, API_BASE, params(seed=2**32 - 1, n=2)))
    assert [j.seed for j in engine.jobs] == [2**32 - 1, 0]


def test_iter_generate_single_image_is_one_request(sdk):
    client, engine = sdk
    results = list(iter_generate(client, MODEL, API_BASE, params(seed=3, n=1)))
    assert len(results) == 1 and results[0].meta["done"] == 1
    assert [(j.n, j.seed) for j in engine.jobs] == [(1, 3)]


def test_iter_generate_stops_between_images(sdk):
    client, engine = sdk
    stop = False
    it = iter_generate(client, MODEL, API_BASE, params(seed=1, n=3), should_stop=lambda: stop)
    first = next(it)
    assert len(first.images) == 1
    stop = True
    last = next(it)
    assert last.meta["cancelled"] is True and last.meta["done"] == 1
    assert len(last.images) == 1
    with pytest.raises(StopIteration):
        next(it)
    assert len(engine.jobs) == 1


def test_iter_generate_cancelled_while_rendering_ends_with_that_image(sdk):
    client, engine = sdk
    # cancel arrives while image 1 is rendering: report it as cancelled, no "running" flicker
    results = list(
        iter_generate(
            client, MODEL, API_BASE, params(seed=1, n=3), should_stop=lambda: len(engine.jobs) >= 1
        )
    )
    assert len(results) == 1
    assert results[0].meta["cancelled"] is True and results[0].meta["done"] == 1
    assert len(engine.jobs) == 1


class TestProgressText:
    def test_running(self):
        assert progress_text({"n": 4, "done": 1}) == "⏳ 生成中 · 已完成 1/4"
        assert progress_text({"n": 1, "done": 0}) == "⏳ 生成中 · 已完成 0/1"

    def test_cancelled(self):
        assert progress_text({"n": 4, "done": 2, "cancelled": True}) == "⏹ 已取消 · 完成 2/4"

    def test_done_with_timings(self):
        meta = {"n": 2, "done": 2, "elapsed_server_seconds": 0.5, "elapsed_client_seconds": 0.75}
        assert progress_text(meta) == "✅ 完成 · 2 张 · 服务端 0.5s / 总 0.8s"

    def test_done_without_timings(self):
        assert (
            progress_text({"n": 1, "done": 1, "elapsed_server_seconds": None}) == "✅ 完成 · 1 张"
        )


def test_iter_edit_streams(sdk, tmp_path: Path):
    client, engine = sdk
    path = tmp_path / "a.png"
    path.write_bytes(png_bytes(64, 32))
    results = list(iter_edit(client, MODEL, API_BASE, [str(path)], params(size=None, seed=9, n=2)))
    assert [len(r.images) for r in results] == [1, 2]
    assert results[-1].meta["seeds"] == [9, 10] and results[-1].meta["refs"] == 1
    assert all(j.n == 1 and len(j.images) == 1 for j in engine.jobs)
    assert "image[]=@" in results[-1].curl


def test_cancel_request_unknown_id_is_false(sdk):
    client, _ = sdk
    assert cancel_request(client, "nothing-running") is False


def test_cancel_request_cancels_in_flight_generation():
    from tests.test_api_cancel import BlockingEngine

    engine = BlockingEngine()
    app = create_app(Settings(_env_file=None, ui=False), engine=engine)
    box: dict = {}
    with TestClient(app) as tc:
        client = openai.OpenAI(base_url=API_BASE, api_key="local", max_retries=0, http_client=tc)

        def worker():
            try:
                run_generate(client, MODEL, API_BASE, params(), request_id="ui-1")
            except openai.APIStatusError as exc:
                box["exc"] = exc

        thread = threading.Thread(target=worker)
        thread.start()
        assert engine.started.wait(5)
        assert cancel_request(client, "ui-1") is True
        thread.join(5)
        engine.release.set()
    assert box["exc"].status_code == 409
    assert error_code(box["exc"]) == "cancelled"


def test_error_code_without_body():
    assert error_code(ValueError("x")) is None


class TestMount:
    def test_ui_enabled(self):
        app = create_app(Settings(_env_file=None, ui=True), engine=FakeEngine())
        with TestClient(app) as tc:
            r = tc.get("/ui", follow_redirects=True)
            assert r.status_code == 200
            assert "gradio" in r.text.lower()

            r = tc.get("/", follow_redirects=False)
            assert r.status_code in (302, 307)
            assert r.headers["location"].endswith("/ui")

            # API still works alongside the mounted UI
            assert tc.get("/health").status_code == 200
            assert tc.get("/v1/models").json()["data"][0]["id"] == MODEL

    def test_ui_disabled(self):
        app = create_app(Settings(_env_file=None, ui=False), engine=FakeEngine())
        with TestClient(app) as tc:
            r = tc.get("/", follow_redirects=False)
            assert r.status_code == 404
            assert r.json()["error"]["code"] == "not_found"
            assert tc.get("/ui").status_code == 404
