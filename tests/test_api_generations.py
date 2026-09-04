import base64
from io import BytesIO

import PIL.Image
import pytest


def _post(client, **body):
    return client.post("/v1/images/generations", json=body)


def _decode(item: dict) -> PIL.Image.Image:
    return PIL.Image.open(BytesIO(base64.b64decode(item["b64_json"])))


def test_happy_path(client, engine):
    r = _post(client, prompt="a red fox", size="512x512", seed=7)
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body) >= {"created", "data", "output_format", "size", "quality", "background"}
    assert body["output_format"] == "png"
    assert body["size"] == "512x512"
    assert body["quality"] is None
    assert body["background"] == "opaque"
    assert isinstance(body["created"], int)
    assert len(body["data"]) == 1
    item = body["data"][0]
    assert item["seed"] == 7
    assert item["revised_prompt"] is None
    img = _decode(item)
    assert img.format == "PNG"
    assert img.size == (512, 512)
    assert float(r.headers["X-Generation-Seconds"]) == pytest.approx(0.25)
    job = engine.jobs[0]
    assert (job.steps, job.width, job.height, job.images, job.n) == (4, 512, 512, None, 1)
    assert job.prompt == "a red fox"


def test_n_and_seed_sequence(client, engine):
    r = _post(client, prompt="x", n=3, seed=10)
    assert r.status_code == 200
    assert [d["seed"] for d in r.json()["data"]] == [10, 11, 12]
    assert engine.jobs[0].n == 3


def test_size_auto(client, engine):
    r = _post(client, prompt="x", size="auto")
    assert r.status_code == 200
    assert r.json()["size"] == "1024x1024"
    assert (engine.jobs[0].width, engine.jobs[0].height) == (None, None)


def test_size_omitted_defaults_to_auto(client, engine):
    r = _post(client, prompt="x")
    assert r.status_code == 200
    assert (engine.jobs[0].width, engine.jobs[0].height) == (None, None)


@pytest.mark.parametrize(
    ("quality", "steps", "echoed"),
    [("high", 8, "high"), ("low", 2, "low"), ("hd", 8, None), (None, 4, None), ("HIGH", 8, "high")],
)
def test_quality_maps_to_steps(client, engine, quality, steps, echoed):
    body = {"prompt": "x"}
    if quality is not None:
        body["quality"] = quality
    r = _post(client, **body)
    assert r.status_code == 200
    assert engine.jobs[0].steps == steps
    assert r.json()["quality"] == echoed


def test_steps_override_quality(client, engine):
    r = _post(client, prompt="x", steps=6, quality="low")
    assert r.status_code == 200
    assert engine.jobs[0].steps == 6


def test_output_format_jpeg(client):
    r = _post(client, prompt="x", output_format="jpeg", output_compression=50, size="256x256")
    assert r.status_code == 200
    assert r.json()["output_format"] == "jpeg"
    assert _decode(r.json()["data"][0]).format == "JPEG"


def test_output_format_webp(client):
    r = _post(client, prompt="x", output_format="webp", size="256x256")
    assert r.status_code == 200
    assert _decode(r.json()["data"][0]).format == "WEBP"


def test_output_format_invalid(client):
    r = _post(client, prompt="x", output_format="gif")
    assert r.status_code == 400
    assert r.json()["error"]["param"] == "output_format"


def test_response_format_url_rejected(client):
    r = _post(client, prompt="x", response_format="url")
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["code"] == "unsupported_parameter"
    assert err["param"] == "response_format"
    assert err["type"] == "invalid_request_error"


def test_response_format_b64_json_ok(client):
    assert _post(client, prompt="x", response_format="b64_json").status_code == 200


def test_stream_rejected(client):
    r = _post(client, prompt="x", stream=True)
    assert r.status_code == 400
    assert r.json()["error"]["param"] == "stream"
    assert r.json()["error"]["code"] == "unsupported_parameter"


def test_stream_false_ok(client):
    assert _post(client, prompt="x", stream=False).status_code == 200


def test_transparent_background_rejected(client):
    r = _post(client, prompt="x", background="transparent")
    assert r.status_code == 400
    assert r.json()["error"]["param"] == "background"


def test_opaque_background_ok(client):
    assert _post(client, prompt="x", background="opaque").status_code == 200


def test_n_too_large(client):
    r = _post(client, prompt="x", n=11)
    assert r.status_code == 400
    assert r.json()["error"]["param"] == "n"
    assert r.json()["error"]["code"] == "invalid_value"


def test_n_zero(client):
    r = _post(client, prompt="x", n=0)
    assert r.status_code == 400
    assert r.json()["error"]["param"] == "n"


def test_missing_prompt(client):
    r = _post(client, n=1)
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["type"] == "invalid_request_error"
    assert err["param"] == "prompt"
    assert err["code"] == "invalid_value"


def test_blank_prompt(client):
    r = _post(client, prompt="   ")
    assert r.status_code == 400
    assert r.json()["error"]["param"] == "prompt"


def test_prompt_is_stripped(client, engine):
    assert _post(client, prompt="  hi  ").status_code == 200
    assert engine.jobs[0].prompt == "hi"


def test_bad_aspect_ratio(client):
    r = _post(client, prompt="x", size="2048x256")
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_aspect_ratio"
    assert r.json()["error"]["param"] == "size"


def test_malformed_size(client):
    r = _post(client, prompt="x", size="big")
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_size"


def test_size_rounded_to_16(client, engine):
    r = _post(client, prompt="x", size="1010x600")
    assert r.status_code == 200
    assert r.json()["size"] == "1008x608"
    assert (engine.jobs[0].width, engine.jobs[0].height) == (1008, 608)


def test_negative_seed(client):
    r = _post(client, prompt="x", seed=-1)
    assert r.status_code == 400
    assert r.json()["error"]["param"] == "seed"


def test_steps_zero(client):
    r = _post(client, prompt="x", steps=0)
    assert r.status_code == 400
    assert r.json()["error"]["param"] == "steps"


def test_invalid_json_body(client):
    r = client.post(
        "/v1/images/generations", content=b"{not json", headers={"content-type": "application/json"}
    )
    assert r.status_code == 400
    assert r.json()["error"]["type"] == "invalid_request_error"


def test_unknown_fields_ignored(client):
    assert _post(client, prompt="x", foo=1).status_code == 200


def test_foreign_model_name_accepted(client):
    assert _post(client, prompt="x", model="gpt-image-1").status_code == 200


def test_generation_runs_off_event_loop(client, engine):
    assert _post(client, prompt="x").status_code == 200
    assert engine.saw_running_loop == [False]


def test_list_models(client):
    r = client.get("/v1/models")
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "list"
    assert [m["id"] for m in body["data"]] == ["flux.2-klein-4b"]
    assert body["data"][0]["object"] == "model"


def test_get_model(client):
    r = client.get("/v1/models/flux.2-klein-4b")
    assert r.status_code == 200
    assert r.json()["id"] == "flux.2-klein-4b"


def test_get_unknown_model(client):
    r = client.get("/v1/models/nope")
    assert r.status_code == 404
    err = r.json()["error"]
    assert err["type"] == "invalid_request_error"
    assert "nope" in err["message"]


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["ready"] is True
    assert body["device"] == "cpu"
    assert isinstance(body["uptime_seconds"], int)


def test_unknown_route(client):
    r = client.get("/nope")
    assert r.status_code == 404
    err = r.json()["error"]
    assert err["type"] == "invalid_request_error"
    assert err["code"] == "not_found"


def test_engine_value_error_becomes_400(client, engine):
    def boom(job):
        raise ValueError("bad job")

    engine.generate = boom
    r = _post(client, prompt="x")
    assert r.status_code == 400
    assert r.json()["error"]["message"] == "bad job"
