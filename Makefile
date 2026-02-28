.DEFAULT_GOAL := help

.PHONY: run test lint setup help

run: ## Start the interactive supervisor chat
	uv run python supervisor.py

test: ## Run unit tests
	uv run pytest tests/ -v

lint: ## Run ruff format + check
	uv run ruff format . && uv run ruff check .

setup: ## Install dependencies
	uv sync

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'
