# agent-registry-router
Registry-driven LLM routing: build classifier prompts from agent descriptions, validate decisions, and dispatch to other agents.

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

## Adapters

- **PydanticAI dispatcher**: `src/agent_registry_router/adapters/pydantic_ai/README.md`

## Tests (uv)

```bash
uv pip install -e ".[dev]"
pytest
```

## Example: FastAPI pinned bypass
See `examples/fastapi_pinned_bypass/`.

## License
Apache-2.0 (see `LICENSE`).
