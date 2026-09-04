import pytest

from flux_server.config import MAX_STEPS, Settings, get_settings


def test_defaults():
    s = Settings(_env_file=None)
    assert s.model == "black-forest-labs/FLUX.2-klein-4B"
    assert s.model_path is None
    assert s.model_id == "flux.2-klein-4b"
    assert (s.device, s.dtype, s.offload, s.weights) == ("auto", "auto", "auto", "auto")
    assert (s.host, s.port) == ("127.0.0.1", 8000)
    assert s.default_steps == 4
    assert s.quality_steps == {
        "low": 2,
        "medium": 4,
        "high": 8,
        "auto": 4,
        "standard": 4,
        "hd": 8,
    }
    assert s.max_n == 10
    assert s.max_pixels == 2048 * 2048
    assert s.max_ref_images == 16
    assert s.max_upload_mb == 32
    assert s.max_in_flight == 8
    assert s.ui is True
    assert s.ui_api_base is None
    assert s.warmup is False
    assert s.local_files_only is True


def test_env_override(monkeypatch):
    monkeypatch.setenv("FLUX_PORT", "9001")
    monkeypatch.setenv("FLUX_DEVICE", "cpu")
    monkeypatch.setenv("FLUX_QUALITY_STEPS", '{"low":1}')
    s = Settings(_env_file=None)
    assert s.port == 9001
    assert s.device == "cpu"
    assert s.quality_steps == {"low": 1}
    assert s.steps_for_quality("low") == 1
    assert s.steps_for_quality("high") == s.default_steps


def test_invalid_device_rejected(monkeypatch):
    monkeypatch.setenv("FLUX_DEVICE", "tpu")
    with pytest.raises(ValueError):
        Settings(_env_file=None)


def test_steps_for_quality():
    s = Settings(_env_file=None)
    assert s.steps_for_quality(None) == 4
    assert s.steps_for_quality("HIGH") == 8
    assert s.steps_for_quality("low") == 2
    assert s.steps_for_quality("bogus") == 4


def test_steps_for_quality_clamped():
    s = Settings(_env_file=None, quality_steps={"low": 0, "high": 999}, default_steps=999)
    assert s.steps_for_quality("low") == 1
    assert s.steps_for_quality("high") == MAX_STEPS
    assert s.steps_for_quality(None) == MAX_STEPS


def test_api_base_default_and_override():
    assert Settings(_env_file=None).api_base == "http://127.0.0.1:8000/v1"
    assert Settings(_env_file=None, port=9001).api_base == "http://127.0.0.1:9001/v1"
    s = Settings(_env_file=None, ui_api_base="http://example.test/v1/")
    assert s.api_base == "http://example.test/v1"


def test_api_base_follows_bind_host():
    assert Settings(_env_file=None, host="192.168.1.5").api_base == "http://192.168.1.5:8000/v1"
    assert Settings(_env_file=None, host="0.0.0.0").api_base == "http://127.0.0.1:8000/v1"
    assert Settings(_env_file=None, host="::", port=81).api_base == "http://127.0.0.1:81/v1"


def test_api_base_ipv6_host_bracketed():
    assert Settings(_env_file=None, host="::1", port=81).api_base == "http://[::1]:81/v1"


def test_get_settings_overrides():
    s = get_settings(_env_file=None, port=1234, warmup=True)
    assert s.port == 1234
    assert s.warmup is True
