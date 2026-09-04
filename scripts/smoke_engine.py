import argparse
import json
import logging
import time
from pathlib import Path

import PIL.Image

from flux_server.config import Settings
from flux_server.engine import Engine, GenerationJob


def parse_size(value: str) -> tuple[int, int]:
    w, h = value.lower().split("x")
    return int(w), int(h)


def main() -> None:
    parser = argparse.ArgumentParser(description="Manual smoke test for flux_server.engine")
    parser.add_argument("--prompt", default="A cat holding a sign that says hello world")
    parser.add_argument("--size", default=None, help="WxH (default 1024x1024 for text-to-image)")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--image", action="append", default=[], help="reference image (repeat)")
    parser.add_argument("--out", default="/tmp/flux-smoke/out.png")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    refs = [PIL.Image.open(p).convert("RGB") for p in args.image] or None
    width = height = None
    if args.size is not None:
        width, height = parse_size(args.size)
    elif not refs:
        width, height = 1024, 1024

    settings = Settings()
    t0 = time.perf_counter()
    engine = Engine.from_settings(settings)
    load_seconds = time.perf_counter() - t0

    result = engine.generate(
        GenerationJob(
            prompt=args.prompt,
            images=refs,
            width=width,
            height=height,
            steps=args.steps,
            seed=args.seed,
            n=args.n,
        )
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, gen in enumerate(result.images):
        path = out if len(result.images) == 1 else out.with_stem(f"{out.stem}_{i}")
        gen.image.save(path)
        paths.append(str(path))

    print(
        json.dumps(
            {
                "info": engine.info(),
                "seeds": [g.seed for g in result.images],
                "elapsed_seconds": round(result.elapsed_seconds, 2),
                "load_seconds": round(load_seconds, 2),
                "size": f"{result.width}x{result.height}",
                "steps": result.steps,
                "outputs": paths,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
