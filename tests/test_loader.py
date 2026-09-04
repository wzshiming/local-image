import json
from pathlib import Path

import pytest
import torch

from flux_server import loader
from flux_server.config import Settings
from flux_server.device import Placement
from flux_server.loader import (
    DOWNLOAD_HINT,
    find_single_file,
    has_diffusers_transformer,
    load_pipeline,
    resolve_snapshot_dir,
    select_weights_source,
)


def make_snapshot(root: Path, *, diffusers: bool, single_file: bool) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "model_index.json").write_text(json.dumps({"_class_name": "Flux2KleinPipeline"}))
    (root / "transformer").mkdir()
    (root / "transformer" / "config.json").write_text("{}")
    if diffusers:
        (root / "transformer" / "diffusion_pytorch_model.safetensors").write_bytes(b"x" * 10)
    if single_file:
        (root / "flux-2-klein-4b.safetensors").write_bytes(b"y" * 20)
    return root


def test_auto_prefers_diffusers(tmp_path):
    snap = make_snapshot(tmp_path, diffusers=True, single_file=True)
    assert has_diffusers_transformer(snap)
    assert select_weights_source(snap) == "diffusers"


def test_auto_falls_back_to_single_file(tmp_path):
    snap = make_snapshot(tmp_path, diffusers=False, single_file=True)
    assert not has_diffusers_transformer(snap)
    assert select_weights_source(snap) == "single_file"


def test_empty_diffusers_file_ignored(tmp_path):
    snap = make_snapshot(tmp_path, diffusers=False, single_file=True)
    (snap / "transformer" / "diffusion_pytorch_model.safetensors").write_bytes(b"")
    assert not has_diffusers_transformer(snap)
    assert select_weights_source(snap) == "single_file"


def test_dangling_symlinks_ignored(tmp_path):
    snap = make_snapshot(tmp_path, diffusers=False, single_file=True)
    (snap / "transformer" / "diffusion_pytorch_model.safetensors").symlink_to(tmp_path / "gone")
    (snap / "broken.safetensors").symlink_to(tmp_path / "gone-too")
    assert not has_diffusers_transformer(snap)
    assert find_single_file(snap) == snap / "flux-2-klein-4b.safetensors"
    assert select_weights_source(snap) == "single_file"


def test_neither_raises_with_hint(tmp_path):
    snap = make_snapshot(tmp_path, diffusers=False, single_file=False)
    with pytest.raises(FileNotFoundError, match="make download"):
        select_weights_source(snap)


def test_explicit_missing_raises(tmp_path):
    only_single = make_snapshot(tmp_path / "a", diffusers=False, single_file=True)
    with pytest.raises(FileNotFoundError, match="make download"):
        select_weights_source(only_single, "diffusers")
    only_diffusers = make_snapshot(tmp_path / "b", diffusers=True, single_file=False)
    with pytest.raises(FileNotFoundError, match="make download"):
        select_weights_source(only_diffusers, "single_file")
    assert select_weights_source(only_single, "single_file") == "single_file"
    assert select_weights_source(only_diffusers, "diffusers") == "diffusers"


def test_find_single_file_picks_largest(tmp_path):
    snap = make_snapshot(tmp_path, diffusers=False, single_file=True)
    (snap / "small.safetensors").write_bytes(b"z" * 5)
    (snap / "big.safetensors").write_bytes(b"z" * 100)
    assert find_single_file(snap) == snap / "big.safetensors"
    assert find_single_file(tmp_path / "missing") is None


def test_resolve_snapshot_dir_model_path(tmp_path):
    snap = make_snapshot(tmp_path / "snap", diffusers=True, single_file=False)
    assert resolve_snapshot_dir(Settings(_env_file=None, model_path=str(snap))) == snap
    with pytest.raises(FileNotFoundError, match="make download"):
        resolve_snapshot_dir(Settings(_env_file=None, model_path=str(tmp_path / "nope")))


def test_resolve_snapshot_dir_uses_snapshot_download(tmp_path, monkeypatch):
    import huggingface_hub

    calls = {}

    def fake_download(repo_id, local_files_only):
        calls["args"] = (repo_id, local_files_only)
        return str(tmp_path)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_download)
    assert resolve_snapshot_dir(Settings(_env_file=None)) == tmp_path
    assert calls["args"] == ("black-forest-labs/FLUX.2-klein-4B", True)

    def failing(repo_id, local_files_only):
        raise OSError("offline")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", failing)
    with pytest.raises(FileNotFoundError, match="make download"):
        resolve_snapshot_dir(Settings(_env_file=None))


class FakePipe:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.device = None

    def to(self, device):
        self.device = device
        return self


class FakePipeline:
    calls: list[tuple[str, dict]] = []

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        cls.calls.append((path, kwargs))
        return FakePipe(path=path, **kwargs)


class FakeTransformer:
    calls: list[tuple[str, dict]] = []

    @classmethod
    def from_single_file(cls, path, **kwargs):
        cls.calls.append((path, kwargs))
        return "TRANSFORMER"


@pytest.fixture
def fake_diffusers(monkeypatch):
    import diffusers

    FakePipeline.calls = []
    FakeTransformer.calls = []
    monkeypatch.setattr(diffusers, "Flux2KleinPipeline", FakePipeline)
    monkeypatch.setattr(diffusers, "Flux2Transformer2DModel", FakeTransformer)
    applied = []
    monkeypatch.setattr(
        loader, "apply_placement", lambda pipe, placement: applied.append((pipe, placement))
    )
    return applied


PLACEMENT = Placement(device="cpu", dtype=torch.float32, offload="none")


def test_load_pipeline_diffusers(tmp_path, fake_diffusers):
    snap = make_snapshot(tmp_path, diffusers=True, single_file=True)
    settings = Settings(_env_file=None, model_path=str(snap))
    loaded = load_pipeline(settings, PLACEMENT)
    assert loaded.weights_source == "diffusers"
    assert loaded.snapshot_dir == snap
    assert loaded.placement is PLACEMENT
    assert FakeTransformer.calls == []
    assert FakePipeline.calls == [(str(snap), {"dtype": torch.float32, "local_files_only": True})]
    assert fake_diffusers == [(loaded.pipe, PLACEMENT)]


def test_load_pipeline_single_file(tmp_path, fake_diffusers):
    snap = make_snapshot(tmp_path, diffusers=False, single_file=True)
    settings = Settings(_env_file=None, model_path=str(snap), weights="single_file")
    loaded = load_pipeline(settings, PLACEMENT)
    assert loaded.weights_source == "single_file"
    assert FakeTransformer.calls == [
        (
            str(snap / "flux-2-klein-4b.safetensors"),
            {
                "config": str(snap),
                "subfolder": "transformer",
                "dtype": torch.float32,
                "local_files_only": True,
            },
        )
    ]
    assert FakePipeline.calls == [
        (
            str(snap),
            {"transformer": "TRANSFORMER", "dtype": torch.float32, "local_files_only": True},
        )
    ]
    assert fake_diffusers == [(loaded.pipe, PLACEMENT)]


def test_download_hint_mentions_make_download():
    assert "make download" in DOWNLOAD_HINT
