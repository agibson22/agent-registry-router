# Contributing

Thanks for your interest in contributing.

## Setup

```bash
git clone https://github.com/agibson22/agent-registry-router.git
cd agent-registry-router
uv venv && source .venv/bin/activate
make install
```

## Development Loop

```bash
make format    # auto-format (ruff + black)
make lint      # lint + typecheck (ruff + black + mypy)
make test      # pytest with coverage gate
```

All three must pass before submitting a PR. CI enforces the same checks.

## Adding an Adapter

1. Create `src/agent_registry_router/adapters/your_framework/`
2. Define duck-typed protocols (no runtime imports of the framework)
3. Implement `route_and_run()` and `route_and_stream()`
4. Add tests with mock agents (no API keys needed)
5. Add an extras group in `pyproject.toml`

See existing adapters (`pydantic_ai`, `openai_agents`, `google_adk`) for the pattern.

## Running the Eval Suite

The eval suite hits external LLM APIs and costs money. It's not part of CI.

```bash
make install-eval
cp .env.example .env  # add API keys
make eval
```

## Pull Requests

- One focused change per PR
- Include tests for new functionality
- Keep coverage above 85%
- Follow existing code style (ruff + black enforce this)
