# agent-registry-router
[![CI](https://img.shields.io/github/actions/workflow/status/agibson22/agent-registry-router/ci.yml?branch=main)](https://github.com/agibson22/agent-registry-router/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/agent-registry-router)](https://pypi.org/project/agent-registry-router/)
[![Python](https://img.shields.io/pypi/pyversions/agent-registry-router)](https://pypi.org/project/agent-registry-router/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

Registry-driven LLM routing: build classifier prompts from agent descriptions, validate decisions, and dispatch to other agents.

## What is this? Why use it?

`agent-registry-router` helps you route LLM traffic to the right agent by:
- Keeping a registry of agents
- Building classifier prompts from that registry
- Validating classifier decisions with clear, typed errors
- Dispatching to runtime agents with optional observability hooks (`on_event`) for logging/metrics.

Use it when you need:
- A stable contract between your classifier and execution layer
- Deterministic, size-aware prompt construction
- Lightweight integration with frameworks like PydanticAI
- Extensibility: plug your own classifier/agents, add hooks for monitoring, and keep routing logic decoupled from business logic

## Features
- Registry-driven classifier prompts (deterministic order, routable-only, size-aware)
- Fail-fast validation with typed errors (no silent fallbacks)
- PydanticAI dispatcher with pinned bypass and observability hooks (`on_event`)
- Typed package (`py.typed`) and CI gates (ruff/black/mypy/pytest-cov)
- FastAPI demo with pinned bypass: `examples/fastapi_pinned_bypass/README.md`

## Scope
- Routing, validation, observability; bring your own LLM clients/agents.
- Adapters are opt-in; core stays lightweight.

## Status
- v0.2.3 — fail-fast by default; adapters namespaced; PydanticAI adapter supports streaming dispatch.

## Install (uv)
From PyPI:

```bash
uv pip install agent-registry-router
```

From a checkout of this repo (dev/editable):

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

## Core usage

```python
from agent_registry_router.core import (
    AgentRegistration,
    AgentRegistry,
    RouteDecision,
    build_classifier_system_prompt,
    validate_route_decision,
)

registry = AgentRegistry()
registry.register(AgentRegistration(name="general", description="General help."))
registry.register(AgentRegistration(name="special", description="Special help."))

prompt = build_classifier_system_prompt(
    registry,
    preamble="You are a query classifier that routes user messages to the appropriate agent.",
    default_agent="general",
)

decision = RouteDecision(agent="special", confidence=0.9, reasoning="Clear match.")
validated = validate_route_decision(decision, registry=registry, default_agent="general")
```

## Behavior & errors

- Default agent must be routable; unknown agents or empty registries raise.
- Classifier selecting a non-routable agent raises `InvalidRouteDecision`.
- Missing default or no routable agents raises `InvalidFallback`.
- Dispatcher raises `AgentNotFound` if the chosen agent cannot be resolved.
- Registry validation uses `RegistryError`; routing errors derive from `RoutingError`.
- Confidence adjustment on invalid routes is unchanged; pinned invalid falls back to the classifier.
- Prompt listing preserves registration order; only routable agents are included.
- Agent descriptions are capped at 512 characters; prompts cannot be built without routable agents. Optional `max_prompt_chars` can bound the generated prompt.
- Observability: `PydanticAIDispatcher` accepts `on_event` callback (receives `RoutingEvent`) and optional logger; emits events for classifier run, validation, pinned bypass, agent resolution, and agent run.

## API contracts

- Public imports (`agent_registry_router.core`): `AgentRegistry`, `AgentRegistration`, `RouteDecision`, `ValidatedRouteDecision`, `validate_route_decision`, `build_classifier_system_prompt`, exceptions (`AgentRegistryRouterError`, `RegistryError`, `RoutingError`, `InvalidRouteDecision`, `InvalidFallback`, `AgentNotFound`), and `RoutingEvent`.
- Adapter: `agent_registry_router.adapters.pydantic_ai` exposes `PydanticAIDispatcher`, `DispatchResult`, `AgentStreamChunk`. Adapters stay namespaced (not re-exported at package root).
- Routing invariants: default agent must be routable; empty registry errors; non-routable selections error; pinned invalid falls back to classifier; classifier output must include `agent` and `confidence` or `InvalidRouteDecision` is raised.
- Prompt determinism: preserves registration order; only routable agents; optional `max_prompt_chars`; description cap 512 chars.
- Hooks: `on_event` receives `RoutingEvent(kind, payload, error)`; hook failures are swallowed; logger is optional.

## Adapters

- **PydanticAI dispatcher**: `src/agent_registry_router/adapters/pydantic_ai/README.md`

## Tests (uv)

```bash
uv pip install -e ".[dev]"
ruff check .
black --check .
mypy --config-file pyproject.mypy.ini .
pytest --cov=agent_registry_router --cov-fail-under=85
```

## Example: FastAPI pinned bypass
See `examples/fastapi_pinned_bypass/`.

## License
Apache-2.0 (see `LICENSE`).
