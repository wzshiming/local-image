import pytest

from flux_server.__main__ import build_parser, settings_from_args


def test_defaults_come_from_settings():
    settings = settings_from_args(build_parser().parse_args([]))
    assert settings.port == 8000
    assert settings.host == "127.0.0.1"
    assert settings.ui is True
    assert settings.warmup is False
    assert settings.device == "auto"
    assert settings.weights == "auto"


def test_flags_override():
    argv = ["--port", "9001", "--device", "cpu", "--no-ui", "--warmup", "--weights", "single_file"]
    settings = settings_from_args(build_parser().parse_args(argv))
    assert settings.port == 9001
    assert settings.device == "cpu"
    assert settings.ui is False
    assert settings.warmup is True
    assert settings.weights == "single_file"


def test_log_level_default():
    assert build_parser().parse_args([]).log_level == "info"
    assert build_parser().parse_args(["--log-level", "debug"]).log_level == "debug"


def test_log_level_rejects_unknown():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--log-level", "trace"])
