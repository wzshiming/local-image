import argparse
import logging

import uvicorn

from flux_server.api import create_app
from flux_server.config import Settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="flux-server",
        description="OpenAI-compatible image API for FLUX.2-klein-4B",
    )
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default=None)
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default=None)
    parser.add_argument("--offload", choices=["auto", "none", "model", "sequential"], default=None)
    parser.add_argument("--weights", choices=["auto", "diffusers", "single_file"], default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--model-path", dest="model_path", default=None)
    parser.add_argument("--no-ui", dest="no_ui", action="store_true", default=None)
    parser.add_argument("--warmup", action="store_true", default=None)
    parser.add_argument(
        "--log-level",
        dest="log_level",
        choices=["critical", "error", "warning", "info", "debug"],
        default="info",
    )
    return parser


def settings_from_args(args: argparse.Namespace) -> Settings:
    overrides: dict = {}
    for key in ("host", "port", "device", "dtype", "offload", "weights", "model", "model_path"):
        value = getattr(args, key, None)
        if value is not None:
            overrides[key] = value
    if args.no_ui:
        overrides["ui"] = False
    if args.warmup:
        overrides["warmup"] = True
    return Settings(**overrides)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    settings = settings_from_args(args)
    app = create_app(settings)
    uvicorn.run(app, host=settings.host, port=settings.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
