# PydanticAI adapter

This adapter provides a small **dispatcher** that runs:

- **pinned session** → skip classifier → dispatch directly
- otherwise → classifier → validate (fallback) → dispatch

It also supports **streaming dispatch** for the selected agent.

It is intentionally **duck-typed** (no hard dependency on `pydantic_ai` imports): any agent object with:

```python
await agent.run(message, deps=...)
```

will work.

## Install

```bash
uv pip install "agent-registry-router[pydanticai]"
```

## API

- `PydanticAIDispatcher(...)`
  - `registry`: `AgentRegistry`
  - `classifier_agent`: classifier agent (PydanticAI-like)
  - `get_agent(name: str) -> agent | None`: resolves agent by name
  - `default_agent`: used as fallback when classifier selects an unknown/non-routable agent

- `await dispatcher.route_and_run(...)`
  - `message: str`
  - `classifier_deps: Any`: passed to the classifier agent
  - `deps_for_agent(agent_name: str) -> Any`: factory for the selected agent deps
  - `pinned_agent: str | None`: if set and resolvable, bypasses classifier (even if not routable)

- `dispatcher.route_and_stream(...) -> AsyncIterator[AgentStreamChunk]`
  - Streams only the **selected agent** (classifier output is never yielded).
  - `stream_classifier: bool = False`: if enabled, consumes a streaming classifier internally to completion, then streams only the chosen agent.

Returns `DispatchResult`:

- `agent_name`: chosen agent name (after validation/fallback)
- `output`: agent run output
- `was_pinned`: whether pinned bypass happened
- `classifier_decision`: raw classifier decision (if classifier ran)
- `validated_decision`: validated decision (if classifier ran)

## Minimal example

```python
from agent_registry_router.adapters.pydantic_ai import PydanticAIDispatcher
from agent_registry_router.core import AgentRegistration, AgentRegistry

registry = AgentRegistry()
registry.register(AgentRegistration(name="general", description="General help."))
registry.register(AgentRegistration(name="special", description="Special help."))

dispatcher = PydanticAIDispatcher(
    registry=registry,
    classifier_agent=classifier_agent,
    get_agent=lambda name: agents.get(name),
    default_agent="general",
)

result = await dispatcher.route_and_run(
    "hello",
    classifier_deps=classifier_deps,
    deps_for_agent=lambda agent_name: deps_for_agent(agent_name),
    pinned_agent=None,
)
print(result.agent_name, result.was_pinned)
```

## Streaming example

```python
from agent_registry_router.adapters.pydantic_ai import PydanticAIDispatcher

async for chunk in dispatcher.route_and_stream(
    "hello",
    classifier_deps=classifier_deps,
    deps_for_agent=lambda agent_name: deps_for_agent(agent_name),
):
    print(chunk.agent_name, chunk.chunk)
```


