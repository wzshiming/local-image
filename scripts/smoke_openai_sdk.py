import argparse
import json
import sys
from pathlib import Path
from typing import Any

from flux_server.ui import GenParams, format_error, make_client, run_edit, run_generate

DEFAULT_REF = Path(
    "~/.cache/huggingface/hub/models--black-forest-labs--FLUX.2-klein-4B/snapshots/"
    "e7b7dc27f91deacad38e78976d1f2b499d76a294/realism.jpg"
).expanduser()


def main() -> int:
    parser = argparse.ArgumentParser(description="End-to-end smoke test via the openai SDK")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--out-dir", default="/tmp/flux-smoke/sdk")
    parser.add_argument("--size", default="512x512")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image", action="append", default=[], help="reference image (repeat)")
    parser.add_argument("--skip-edit", action="store_true")
    args = parser.parse_args()

    api_base = args.base_url.rstrip("/")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    refs = args.image or ([str(DEFAULT_REF)] if DEFAULT_REF.exists() else [])
    summary: dict[str, Any] = {"base_url": api_base, "out_dir": str(out_dir)}

    try:
        client = make_client(api_base)
        root = client.with_options(base_url=api_base.removesuffix("/v1"))
        summary["health"] = root.get("/health", cast_to=dict[str, Any])
        models = [m.id for m in client.models.list().data]
        summary["models"] = models
        model_id = models[0]

        gen = run_generate(
            client,
            model_id,
            api_base,
            GenParams(
                prompt="A cat holding a sign that says hello world",
                size=args.size,
                quality=None,
                steps=args.steps,
                seed=args.seed,
                n=1,
                output_format="png",
            ),
        )
        for img, caption in gen.images:
            img.save(out_dir / f"gen_{caption.removeprefix('seed=')}.png")
        summary["generate"] = gen.meta

        if args.skip_edit or not refs:
            summary["edit"] = "skipped"
        else:
            edit = run_edit(
                client,
                model_id,
                api_base,
                refs,
                GenParams(
                    prompt="make it a snowy winter scene",
                    size=None,
                    quality=None,
                    steps=args.steps,
                    seed=args.seed,
                    n=1,
                    output_format="png",
                ),
            )
            for img, caption in edit.images:
                img.save(out_dir / f"edit_{caption.removeprefix('seed=')}.png")
            summary["edit"] = edit.meta
    except Exception as exc:
        summary["error"] = format_error(exc)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 1

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
