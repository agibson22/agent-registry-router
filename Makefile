PYTHON ?= python

.PHONY: install install-eval lint format typecheck test example eval eval-report

install:
	uv pip install -e ".[dev]"

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy --config-file pyproject.mypy.ini .

format:
	uv run ruff check --fix .
	uv run ruff format .

typecheck:
	uv run mypy --config-file pyproject.mypy.ini .

test:
	uv run pytest --cov=agent_registry_router --cov-fail-under=85

example:
	cd examples/fastapi_pinned_bypass && PYTHONPATH=../../src uv run python -m uvicorn main:app --reload

install-eval:
	uv pip install -e ".[dev,eval]"

eval:
	PYTHONPATH=src uv run python evals/run_eval.py --report $(ARGS)

eval-report:
	uv run python evals/report.py $(ARGS)

