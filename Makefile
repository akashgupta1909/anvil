.DEFAULT_GOAL := help
.PHONY: help install lint format typecheck test check requirements

help:  ## Show targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-14s %s\n", $$1, $$2}'

install:  ## Install the project and dev tools into .venv
	uv sync --frozen

lint:  ## Lint and check formatting
	uv run ruff check .
	uv run ruff format --check .

format:  ## Auto-format and fix lint
	uv run ruff format .
	uv run ruff check --fix .

typecheck:  ## Static types (pyright, strict)
	uv run pyright

test:  ## Unit tests (no network)
	uv run pytest

check: lint typecheck test  ## Everything CI runs




requirements:  ## Regenerate requirements.txt from the lock file
	uv export --frozen --no-dev --no-emit-project --no-hashes -o requirements.txt



