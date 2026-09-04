"""Benchmark generation parameters against a running flux-server instance."""

import argparse
import json
import time
from collections import defaultdict
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_SIZES = ("512x512", "768x768", "1024x1024")
DEFAULT_STEPS = (2, 4)


def parse_sizes(value: str) -> list[str]:
    sizes = [item.strip() for item in value.split(",") if item.strip()]
    for size in sizes:
        parts = size.lower().split("x")
        if len(parts) != 2 or any(not part.isdigit() or int(part) < 16 for part in parts):
            raise argparse.ArgumentTypeError(f"Invalid size: {size!r}; expected WxH")
    return sizes


def parse_steps(value: str) -> list[int]:
    try:
        steps = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("steps must be comma-separated integers") from exc
    if not steps or any(step < 1 for step in steps):
        raise argparse.ArgumentTypeError("steps must contain positive integers")
    return steps


def request_json(url: str, payload: dict | None = None, timeout: float = 3600) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    request = Request(url, data=data, headers=headers, method="POST" if data else "GET")
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def generate(base_url: str, payload: dict, timeout: float) -> dict:
    url = f"{base_url.rstrip('/')}/images/generations"
    data = json.dumps(payload).encode("utf-8")
    request = Request(
        url,
        data=data,
        headers={"Accept": "application/json", "Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urlopen(request, timeout=timeout) as response:
            response.read()
            wall_seconds = time.perf_counter() - started
            generation = response.headers.get("X-Generation-Seconds")
            return {
                "status": response.status,
                "wall_seconds": round(wall_seconds, 3),
                "generation_seconds": float(generation) if generation else None,
            }
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return {
            "status": exc.code,
            "wall_seconds": round(time.perf_counter() - started, 3),
            "error": body[:500],
        }
    except (TimeoutError, URLError, OSError) as exc:
        return {
            "status": None,
            "wall_seconds": round(time.perf_counter() - started, 3),
            "error": str(exc),
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark size/steps combinations against a running flux-server"
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--prompt", default="a red apple on a wooden table, studio photo")
    parser.add_argument("--sizes", type=parse_sizes, default=list(DEFAULT_SIZES))
    parser.add_argument("--steps", type=parse_steps, default=list(DEFAULT_STEPS))
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--repeats", type=int, default=1, help="runs per combination")
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--warmup", action="store_true", help="run a small request first")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")

    base_url = args.base_url.rstrip("/")
    health = request_json(f"{base_url.removesuffix('/v1')}/health", timeout=args.timeout)
    print(json.dumps({"health": health}, ensure_ascii=False))

    if args.warmup:
        print("warmup", flush=True)
        result = generate(
            base_url,
            {"prompt": args.prompt, "size": "256x256", "steps": 1, "seed": args.seed},
            args.timeout,
        )
        print(json.dumps({"warmup": result}, ensure_ascii=False), flush=True)

    results: list[dict] = []
    for size in args.sizes:
        for steps in args.steps:
            for repeat in range(1, args.repeats + 1):
                result = generate(
                    base_url,
                    {
                        "prompt": args.prompt,
                        "size": size,
                        "steps": steps,
                        "seed": args.seed,
                        "output_format": "webp",
                        "output_compression": 80,
                    },
                    args.timeout,
                )
                row = {"size": size, "steps": steps, "repeat": repeat, **result}
                results.append(row)
                print(json.dumps(row, ensure_ascii=False), flush=True)

    successful = [row for row in results if row["status"] == 200]
    by_size: dict[str, list[dict]] = defaultdict(list)
    for row in successful:
        by_size[row["size"]].append(row)
    recommendations = {}
    for size, rows in by_size.items():
        recommendations[size] = min(
            rows, key=lambda row: row["generation_seconds"] or row["wall_seconds"]
        )
    print(
        json.dumps(
            {
                "summary": {
                    "successful": len(successful),
                    "failed": len(results) - len(successful),
                    "recommendations": recommendations,
                }
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
