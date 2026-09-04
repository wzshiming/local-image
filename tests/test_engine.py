import threading
from types import SimpleNamespace

import PIL.Image
import pytest
import torch

from flux_server.config import Settings
from flux_server.device import Placement
from flux_server.engine import Engine, GenerationCancelled, GenerationJob


class FakePipe:
    def __init__(self, size=(64, 64)):
        self.calls: list[dict] = []
        self.size = size

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(images=[PIL.Image.new("RGB", self.size, "red")])


class SteppingPipe(FakePipe):
    """Calls callback_on_step_end per step like the real pipeline; can set cancel mid-way."""

    def __init__(self, cancel=None, cancel_at_step=None, cancel_after_call=False):
        super().__init__()
        self.cancel = cancel
        self.cancel_at_step = cancel_at_step
        self.cancel_after_call = cancel_after_call
        self.steps_done = 0

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        callback = kwargs.get("callback_on_step_end")
        for step in range(kwargs["num_inference_steps"]):
            if step == self.cancel_at_step:
                self.cancel.set()
            if callback is not None:
                callback(self, step, 0, {})
            self.steps_done += 1
        if self.cancel_after_call:
            self.cancel.set()
        return SimpleNamespace(images=[PIL.Image.new("RGB", self.size, "red")])


def make_engine(pipe=None, inpaint_pipe=None, **settings_overrides):
    pipe = pipe or FakePipe()
    settings = Settings(_env_file=None, **settings_overrides)
    placement = Placement(device="cpu", dtype=torch.float32, offload="none")
    engine = Engine(
        pipe,
        settings,
        placement,
        weights_source="diffusers",
        snapshot_dir=None,
        inpaint_pipe=inpaint_pipe,
    )
    return engine, pipe


class FakeInpaintPipe:
    """Returns a red image the size of the source, like the real inpaint pipeline."""

    def __init__(self):
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(images=[PIL.Image.new("RGB", kwargs["image"].size, (255, 0, 0))])


def test_seeds_are_consecutive_and_match_generators():
    engine, pipe = make_engine()
    result = engine.generate(GenerationJob(prompt="hi", seed=5, n=3, steps=2))
    assert [g.seed for g in result.images] == [5, 6, 7]
    assert [c["generator"].initial_seed() for c in pipe.calls] == [5, 6, 7]
    assert all(c["generator"].device.type == "cpu" for c in pipe.calls)
    assert result.steps == 2
    assert len(pipe.calls) == 3


def test_random_seed_when_none():
    engine, pipe = make_engine()
    result = engine.generate(GenerationJob(prompt="hi", n=2))
    seeds = [g.seed for g in result.images]
    assert 0 <= seeds[0] < 2**31
    assert seeds[1] == seeds[0] + 1
    assert [c["generator"].initial_seed() for c in pipe.calls] == seeds


def test_seeds_wrap_at_32_bits():
    engine, pipe = make_engine()
    result = engine.generate(GenerationJob(prompt="hi", seed=2**32 - 1, n=2, steps=1))
    assert [g.seed for g in result.images] == [2**32 - 1, 0]
    assert [c["generator"].initial_seed() for c in pipe.calls] == [2**32 - 1, 0]


def test_pipe_kwargs_text_to_image():
    engine, pipe = make_engine()
    engine.generate(GenerationJob(prompt="a cat", width=256, height=128, steps=4, seed=1))
    call = pipe.calls[0]
    assert call["prompt"] == "a cat"
    assert call["image"] is None
    assert (call["width"], call["height"]) == (256, 128)
    assert call["num_inference_steps"] == 4
    assert call["guidance_scale"] == 1.0
    assert call["num_images_per_prompt"] == 1
    assert call["output_type"] == "pil"


def test_pipe_kwargs_edit_passes_reference_images():
    engine, pipe = make_engine()
    refs = [PIL.Image.new("RGB", (32, 32)), PIL.Image.new("RGB", (16, 16))]
    engine.generate(GenerationJob(prompt="edit", images=refs, seed=1))
    call = pipe.calls[0]
    assert call["image"] is refs
    assert call["width"] is None and call["height"] is None


def test_empty_images_list_becomes_none():
    engine, pipe = make_engine()
    engine.generate(GenerationJob(prompt="edit", images=[], seed=1))
    assert pipe.calls[0]["image"] is None


def test_invalid_args_raise():
    engine, _ = make_engine()
    with pytest.raises(ValueError):
        engine.generate(GenerationJob(prompt="x", steps=0))
    with pytest.raises(ValueError):
        engine.generate(GenerationJob(prompt="x", n=0))


def test_result_size_from_image():
    engine, _ = make_engine(FakePipe(size=(48, 80)))
    result = engine.generate(GenerationJob(prompt="x", seed=0))
    assert (result.width, result.height) == (48, 80)
    assert result.elapsed_seconds >= 0
    assert result.images[0].image.size == (48, 80)


def test_no_cancel_event_means_no_callback():
    engine, pipe = make_engine(SteppingPipe())
    engine.generate(GenerationJob(prompt="x", seed=0, steps=2))
    assert pipe.calls[0]["callback_on_step_end"] is None
    assert pipe.steps_done == 2


def test_cancel_before_start_skips_pipeline():
    cancel = threading.Event()
    cancel.set()
    engine, pipe = make_engine(SteppingPipe())
    with pytest.raises(GenerationCancelled):
        engine.generate(GenerationJob(prompt="x", seed=0, cancel=cancel))
    assert pipe.calls == []


def test_cancel_mid_generation_stops_at_step_boundary():
    cancel = threading.Event()
    engine, pipe = make_engine(SteppingPipe(cancel=cancel, cancel_at_step=1))
    with pytest.raises(GenerationCancelled):
        engine.generate(GenerationJob(prompt="x", seed=0, steps=4, cancel=cancel))
    assert pipe.steps_done == 1
    assert pipe.calls[0]["callback_on_step_end"] is not None


def test_cancel_between_images_skips_remaining():
    cancel = threading.Event()
    engine, pipe = make_engine(SteppingPipe(cancel=cancel, cancel_after_call=True))
    with pytest.raises(GenerationCancelled):
        engine.generate(GenerationJob(prompt="x", seed=0, steps=2, n=3, cancel=cancel))
    assert len(pipe.calls) == 1


def test_lock_released_after_cancel():
    cancel = threading.Event()
    cancel.set()
    engine, pipe = make_engine(SteppingPipe())
    with pytest.raises(GenerationCancelled):
        engine.generate(GenerationJob(prompt="x", seed=0, cancel=cancel))
    result = engine.generate(GenerationJob(prompt="x", seed=0, steps=1))
    assert len(result.images) == 1


def test_strength_only_uses_inpaint_pipe_with_full_mask():
    inpaint = FakeInpaintPipe()
    engine, pipe = make_engine(inpaint_pipe=inpaint)
    source = PIL.Image.new("RGB", (64, 32), (0, 0, 255))
    result = engine.generate(GenerationJob(prompt="x", images=[source], strength=0.5, seed=1))
    assert pipe.calls == []
    call = inpaint.calls[0]
    assert call["image"] is source
    assert call["image_reference"] is None
    assert call["strength"] == 0.5
    assert call["mask_image"].mode == "L" and call["mask_image"].size == (64, 32)
    assert call["mask_image"].getextrema() == (255, 255)
    assert call["guidance_scale"] == 1.0 and call["num_inference_steps"] == 4
    assert "width" not in call and "height" not in call
    assert result.images[0].image.getpixel((0, 0)) == (255, 0, 0)  # no compositing without mask
    assert (result.width, result.height) == (64, 32)


def test_mask_with_reference_composites_source_back():
    inpaint = FakeInpaintPipe()
    engine, _ = make_engine(inpaint_pipe=inpaint)
    source = PIL.Image.new("RGB", (8, 4), (0, 0, 255))
    ref = PIL.Image.new("RGB", (16, 16), "green")
    mask = PIL.Image.new("L", (8, 4), 0)
    for x in range(4):
        for y in range(4):
            mask.putpixel((x, y), 255)
    result = engine.generate(GenerationJob(prompt="x", images=[source, ref], mask=mask, seed=1))
    call = inpaint.calls[0]
    assert call["mask_image"] is mask
    assert call["image_reference"] is ref
    assert call["strength"] == 1.0  # default: fully repaint the masked area
    out = result.images[0].image
    assert out.getpixel((0, 0)) == (255, 0, 0)  # repainted
    assert out.getpixel((7, 3)) == (0, 0, 255)  # kept from the source


def test_inpaint_validation():
    engine, _ = make_engine(inpaint_pipe=FakeInpaintPipe())
    img = PIL.Image.new("RGB", (8, 8))
    with pytest.raises(ValueError, match="source image"):
        engine.generate(GenerationJob(prompt="x", strength=0.5))
    with pytest.raises(ValueError, match="at most one"):
        engine.generate(GenerationJob(prompt="x", images=[img, img, img], strength=0.5))
    for bad in (0.0, 1.5):
        with pytest.raises(ValueError, match="strength"):
            engine.generate(GenerationJob(prompt="x", images=[img], strength=bad))


def test_inpaint_pipe_built_lazily_once(monkeypatch):
    built: list[object] = []

    def fake_build(pipe):
        built.append(pipe)
        return FakeInpaintPipe()

    monkeypatch.setattr("flux_server.engine.build_inpaint_pipe", fake_build)
    engine, pipe = make_engine()
    img = PIL.Image.new("RGB", (8, 8))
    engine.generate(GenerationJob(prompt="x", images=[img], seed=0))  # native path: no build
    assert built == []
    engine.generate(GenerationJob(prompt="x", images=[img], strength=0.5, seed=0))
    engine.generate(GenerationJob(prompt="x", images=[img], strength=0.7, seed=0))
    assert built == [pipe]


def test_inpaint_cancel_callback_passed():
    inpaint = FakeInpaintPipe()
    engine, _ = make_engine(inpaint_pipe=inpaint)
    cancel = threading.Event()
    img = PIL.Image.new("RGB", (8, 8))
    engine.generate(GenerationJob(prompt="x", images=[img], strength=0.5, cancel=cancel, seed=0))
    assert inpaint.calls[0]["callback_on_step_end"] is not None


def test_info_keys():
    engine, _ = make_engine(model_id="custom-id", default_steps=6)
    info = engine.info()
    assert info == {
        "model": "black-forest-labs/FLUX.2-klein-4B",
        "model_id": "custom-id",
        "device": "cpu",
        "dtype": "float32",
        "offload": "none",
        "weights_source": "diffusers",
        "snapshot_dir": None,
        "default_steps": 6,
        "ready": True,
    }
    assert engine.model_id == "custom-id"
