PYTHON ?= python

.PHONY: install lint format typecheck test example

install:
	uv pip install -e ".[dev]"

lint:
	uv run ruff check .
	uv run black --check .
	uv run mypy --config-file pyproject.mypy.ini .

format:
	uv run ruff check --fix .
	uv run black .

typecheck:
	uv run mypy --config-file pyproject.mypy.ini .

test:
	uv run pytest --cov=agent_registry_router --cov-fail-under=85

example:
	cd examples/fastapi_pinned_bypass && PYTHONPATH=../../src uv run python -m uvicorn main:app --reload

