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

## Streaming modes

This adapter supports two PydanticAI-native streaming modes for the **selected agent**:

- **Text streaming**: `dispatcher.route_and_stream(...) -> AsyncIterator[AgentStreamChunk]`
  - Streams text chunks via the agent stream result’s `stream_text(...)`.
  - Classifier output is never yielded; only the selected agent is streamed.

- **Response streaming**: `dispatcher.route_and_stream_responses(...) -> ResponseStreamSession`
  - Returns an async context manager that exposes the underlying streamed run handle (`session.streamed_run`).
  - Streams `(model_response, is_last)` items via `session.iter_responses()`.
  - This is ideal if you want to do framework-native progressive validation (e.g. `validate_response_output(..., allow_partial=...)`) and/or extract usage/tool messages from the run handle.

Both support:
- `pinned_agent: str | None`: if set and resolvable, bypasses the classifier (even if not routable)
- `stream_classifier: bool = False`: if enabled, consumes a streaming classifier internally to completion (classifier streaming is never yielded)
- Streaming classifiers **must emit a final decision/output** (e.g., a result event or final streamed item); otherwise the dispatcher raises `InvalidRouteDecision`.

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

## Response streaming example (PydanticAI model responses)

```python
from agent_registry_router.adapters.pydantic_ai import PydanticAIDispatcher

async with dispatcher.route_and_stream_responses(
    "hello",
    classifier_deps=classifier_deps,
    deps_for_agent=lambda agent_name: deps_for_agent(agent_name),
) as session:
    async for item in session.iter_responses():
        model_response = item.model_response
        is_last = item.is_last

        # If your framework exposes progressive validation, do it here.
        # Example (PydanticAI-style API):
        # partial = await session.streamed_run.validate_response_output(
        #     model_response,
        #     allow_partial=not is_last,
        # )
        print(session.agent_name, is_last, model_response)
```


