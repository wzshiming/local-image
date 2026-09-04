export UV_PROJECT_ENVIRONMENT := .venv
.DEFAULT_GOAL := help

.PHONY: help sync sync-cuda run test lint fmt download clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-10s %s\n", $$1, $$2}'

sync: ## Create .venv and install CPU PyTorch
	uv sync --all-groups --upgrade-package torch

sync-cuda: ## Create .venv and install CUDA-enabled PyTorch
	uv sync --all-groups --upgrade-package torch --index https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match

run: ## Start the server without changing the selected PyTorch build
	uv run --no-sync python -m flux_server

test: ## Run pytest
	uv run pytest

lint: ## Run ruff check + format check
	uv run ruff check src tests && uv run ruff format --check src tests

fmt: ## Format and auto-fix with ruff
	uv run ruff format src tests && uv run ruff check --fix src tests

download: ## Download FLUX.2-klein-4B weights into the Hugging Face cache
	uv run hf download black-forest-labs/FLUX.2-klein-4B

clean: ## Remove caches (keeps the venv)
	rm -rf .pytest_cache .ruff_cache dist build
	find . -path ./.venv -prune -o -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
	find . -path ./.venv -prune -o -name '*.egg-info' -type d -exec rm -rf {} + 2>/dev/null || true
