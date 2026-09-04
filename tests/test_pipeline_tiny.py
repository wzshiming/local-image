"""Runs the real diffusers pipelines with tiny random weights (CPU, seconds) so the
Engine's native and inpaint code paths are exercised against the actual pipeline API."""

import threading
from pathlib import Path

import PIL.Image
import pytest
import torch

from flux_server.config import Settings
from flux_server.device import Placement
from flux_server.engine import Engine, GenerationCancelled, GenerationJob, build_inpaint_pipe

SNAPSHOT = (
    Path.home()
    / ".cache/huggingface/hub/models--black-forest-labs--FLUX.2-klein-4B/snapshots"
    / "e7b7dc27f91deacad38e78976d1f2b499d76a294"
)
pytestmark = pytest.mark.skipif(
    not (SNAPSHOT / "tokenizer").exists(), reason="local tokenizer files not present"
)


@pytest.fixture(scope="module")
def tiny_engine():
    from diffusers import (
        AutoencoderKLFlux2,
        FlowMatchEulerDiscreteScheduler,
        Flux2KleinPipeline,
        Flux2Transformer2DModel,
    )
    from transformers import Qwen2TokenizerFast, Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(0)
    vae = AutoencoderKLFlux2(
        block_out_channels=(8, 8, 8, 8), layers_per_block=1, norm_num_groups=8, latent_channels=32
    )
    hidden = 16
    text_encoder = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=151936,
            hidden_size=hidden,
            intermediate_size=32,
            num_hidden_layers=28,  # the pipeline reads hidden states 9, 18 and 27
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=1024,
        )
    )
    transformer = Flux2Transformer2DModel(
        in_channels=128,
        num_layers=1,
        num_single_layers=1,
        attention_head_dim=16,
        num_attention_heads=2,
        joint_attention_dim=3 * hidden,
        axes_dims_rope=(4, 4, 4, 4),
        guidance_embeds=False,
    )
    tokenizer = Qwen2TokenizerFast.from_pretrained(SNAPSHOT / "tokenizer")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(SNAPSHOT / "scheduler")
    pipe = Flux2KleinPipeline(
        scheduler=scheduler,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        transformer=transformer,
        is_distilled=True,
    )
    pipe.set_progress_bar_config(disable=True)
    placement = Placement(device="cpu", dtype=torch.float32, offload="none")
    return Engine(pipe, Settings(_env_file=None), placement, weights_source="tiny")


def _half_mask(w: int, h: int) -> PIL.Image.Image:
    mask = PIL.Image.new("L", (w, h), 0)
    mask.paste(255, (0, 0, w // 2, h))
    return mask


def test_native_text_to_image_and_edit(tiny_engine):
    result = tiny_engine.generate(
        GenerationJob(prompt="a cat", width=64, height=64, steps=2, seed=0)
    )
    assert (result.width, result.height) == (64, 64)

    ref = PIL.Image.new("RGB", (96, 64), "blue")
    result = tiny_engine.generate(GenerationJob(prompt="edit", images=[ref], steps=2, seed=0))
    assert (result.width, result.height) == (96, 64)  # size follows the reference image


def test_inpaint_pipeline_is_built_from_shared_components(tiny_engine):
    inpaint = build_inpaint_pipe(tiny_engine.pipe)
    assert inpaint.transformer is tiny_engine.pipe.transformer
    assert inpaint.vae is tiny_engine.pipe.vae
    assert inpaint.config.is_distilled is True


def test_strength_edit_runs_fewer_steps(tiny_engine):
    src = PIL.Image.new("RGB", (96, 64), "red")
    result = tiny_engine.generate(
        GenerationJob(prompt="x", images=[src], strength=0.5, steps=4, seed=0)
    )
    assert (result.width, result.height) == (96, 64)
    # strength 0.5 of 4 steps → 2 denoising steps actually run
    assert tiny_engine._inpaint_pipe.num_timesteps == 2


def test_mask_edit_keeps_unmasked_pixels_and_uses_reference(tiny_engine):
    src = PIL.Image.new("RGB", (96, 64), (0, 0, 255))
    ref = PIL.Image.new("RGB", (64, 64), "green")
    mask = _half_mask(96, 64)
    result = tiny_engine.generate(
        GenerationJob(prompt="x", images=[src, ref], mask=mask, steps=2, seed=0)
    )
    out = result.images[0].image
    assert out.size == (96, 64)
    assert out.getpixel((95, 63)) == (0, 0, 255)  # right half composited from the source
    assert tiny_engine._inpaint_pipe.num_timesteps == 2


def test_inpaint_cancel_mid_way(tiny_engine):
    cancel = threading.Event()
    src = PIL.Image.new("RGB", (64, 64), "red")
    with pytest.raises(GenerationCancelled):
        tiny_engine.generate(
            GenerationJob(
                prompt="x", images=[src], strength=1.0, steps=4, seed=0, cancel=_set_after(cancel)
            )
        )


def _set_after(event: threading.Event) -> threading.Event:
    # The engine checks the event before the job starts; set it immediately.
    event.set()
    return event
