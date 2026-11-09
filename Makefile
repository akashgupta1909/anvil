.DEFAULT_GOAL := help
.PHONY: help install lint format typecheck test check run serve eval requirements docker-build docker-up docker-down

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

run:  ## Run one task: make run TASK="list the files"
	uv run anvil run "$(TASK)"

serve:  ## Start the API on :8000
	uv run anvil serve --host 0.0.0.0 --port 8000

eval:  ## Evaluation suite: make eval ARGS="--ablation baseline --ablation planner"
	uv run anvil eval $(ARGS)

requirements:  ## Regenerate requirements.txt from the lock file
	uv export --frozen --no-dev --no-emit-project --no-hashes -o requirements.txt

docker-build:  ## Build the image
	docker build -t anvil:latest .

docker-up:  ## Start the stack (needs .env)
	docker compose up -d --build

docker-down:  ## Stop the stack
	docker compose down
