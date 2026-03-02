from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import pytest
from pydantic import BaseModel

from agent_registry_router.adapters.pydantic_ai import (
    AgentResponseStreamItem,
    AgentStreamChunk,
    DispatchResult,
    PydanticAIDispatcher,
)
from agent_registry_router.core import (
    AgentNotFound,
    AgentRegistration,
    AgentRegistry,
    InvalidRouteDecision,
    RouteDecision,
    RoutingEvent,
)


@dataclass
class FakeRunResult:
    output: Any


class FakeAgent:
    def __init__(self, output: Any):
        self._output = output
        self.called = False
        self.last_deps: Any | None = None

    async def run(self, message: str, *, deps: Any) -> FakeRunResult:
        self.called = True
        self.last_deps = deps
        return FakeRunResult(self._output)


class FakeStreamedRunResult:
    def __init__(self, chunks: list[str]):
        self._chunks = chunks

    async def stream_text(self, *, delta: bool = False) -> AsyncIterator[str]:  # noqa: ARG002
        for chunk in self._chunks:
            yield chunk


class FakeResponseStreamedRunResult:
    def __init__(self, responses: list[tuple[Any, bool]]):
        self._responses = responses

    async def stream_responses(
        self, *, debounce_by: float | None = None
    ) -> AsyncIterator[tuple[Any, bool]]:  # noqa: ARG002
        for item in self._responses:
            yield item


class FakeRunStreamContextManager:
    def __init__(self, streamed: Any):
        self._streamed = streamed

    async def __aenter__(self) -> Any:
        return self._streamed

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:  # noqa: ANN401
        return None


class FakeStreamingAgent:
    def __init__(self, chunks: list[str]):
        self._streamed = FakeStreamedRunResult(chunks)
        self.called = False
        self.last_deps: Any | None = None

    def run_stream(self, message: str, *, deps: Any) -> FakeRunStreamContextManager:  # noqa: ARG002
        self.called = True
        self.last_deps = deps
        return FakeRunStreamContextManager(self._streamed)


class FakeResponseStreamingAgent:
    def __init__(self, responses: list[tuple[Any, bool]]):
        self._streamed = FakeResponseStreamedRunResult(responses)
        self.called = False
        self.last_deps: Any | None = None

    def run_stream(self, message: str, *, deps: Any) -> FakeRunStreamContextManager:  # noqa: ARG002
        self.called = True
        self.last_deps = deps
        return FakeRunStreamContextManager(self._streamed)


class FakeStreamingAgentMissingStreamResponses:
    def __init__(self) -> None:
        self._streamed = FakeStreamedRunResult(["x"])

    def run_stream(self, message: str, *, deps: Any) -> FakeRunStreamContextManager:  # noqa: ARG002
        return FakeRunStreamContextManager(self._streamed)


class FakeStreamedRunResultNoDelta:
    def __init__(self, chunks: list[str]):
        self._chunks = chunks

    async def stream_text(self) -> AsyncIterator[str]:
        for chunk in self._chunks:
            yield chunk


class FakeStreamingAgentNoDelta:
    def __init__(self, chunks: list[str]):
        self._streamed = FakeStreamedRunResultNoDelta(chunks)
        self.called = False

    def run_stream(self, message: str, *, deps: Any) -> FakeRunStreamContextManager:  # noqa: ARG002
        self.called = True
        return FakeRunStreamContextManager(self._streamed)  # type: ignore[arg-type]


class FakeRunStreamNoTextContextManager:
    async def __aenter__(self) -> object:
        return object()

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:  # noqa: ANN401
        return None


class FakeStreamingAgentMissingStreamText:
    def run_stream(
        self, message: str, *, deps: Any
    ) -> FakeRunStreamNoTextContextManager:  # noqa: ARG002
        return FakeRunStreamNoTextContextManager()


@dataclass
class FakeClassifierRunResultEvent:
    result: FakeRunResult


class FakeStreamingClassifier:
    def __init__(self, output: Any, *, include_final_event: bool = True):
        self._output = output
        self._include_final_event = include_final_event

    async def run(self, message: str, *, deps: Any) -> FakeRunResult:  # noqa: ARG002
        # Fallback non-streaming path; return the same output for convenience.
        return FakeRunResult(self._output)

    async def run_stream_events(
        self, message: str, *, deps: Any
    ) -> AsyncIterator[Any]:  # noqa: ARG002
        yield object()  # some non-result event
        if self._include_final_event:
            yield FakeClassifierRunResultEvent(result=FakeRunResult(self._output))


class FakeClassifierStreamedRunResultOutputSync:
    def __init__(self, items: list[Any]):
        self._items = items

    def stream_output(self) -> Any:
        return iter(self._items)


class FakeClassifierStreamedRunResultOutputAsync:
    def __init__(self, items: list[Any]):
        self._items = items

    async def _aiter(self) -> AsyncIterator[Any]:
        for item in self._items:
            yield item

    def stream_output(self) -> Any:
        return self._aiter()


class FakeClassifierStreamedRunResultTextNoDelta:
    def __init__(self, items: list[Any]):
        self._items = items

    async def stream_text(self) -> AsyncIterator[Any]:
        for item in self._items:
            yield item


class FakeClassifierStreamedRunResultNoStreams:
    pass


class FakeClassifierStreamedRunResultNoOutput:
    def stream_output(self) -> Any:
        return iter(())


class FakeClassifierRunStreamContextManager:
    def __init__(self, streamed: Any):
        self._streamed = streamed

    async def __aenter__(self) -> Any:
        return self._streamed

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:  # noqa: ANN401
        return None


class FakeStreamingClassifierRunStream:
    def __init__(self, streamed: Any, *, output: Any | None = None):
        self._streamed = streamed
        self._output = output

    async def run(self, message: str, *, deps: Any) -> FakeRunResult:  # noqa: ARG002
        return FakeRunResult(self._output)

    def run_stream(
        self, message: str, *, deps: Any
    ) -> FakeClassifierRunStreamContextManager:  # noqa: ARG002
        return FakeClassifierRunStreamContextManager(self._streamed)


def test_dispatcher_classifier_output_as_basemodel_is_accepted() -> None:
    registry = AgentRegistry()
    registry.register(AgentRegistration(name="general", description="General help."))
    registry.register(AgentRegistration(name="special", description="Special help."))

    class DecisionModel(BaseModel):
        agent: str
        confidence: float
        reasoning: str | None = None

    classifier = FakeAgent(DecisionModel(agent="special", confidence=0.9, reasoning="Ok"))
    special = FakeAgent({"answer": "from special"})
    agents: dict[str, FakeAgent] = {
        "special": special,
        "general": FakeAgent({"answer": "from general"}),
    }

    dispatcher = PydanticAIDispatcher(
        registry=registry,
        classifier_agent=classifier,
        get_agent=lambda name: agents.get(name),
        default_agent="general",
    )

    result = _run(dispatcher)
    assert result.agent_name == "special"


def test_dispatcher_classifier_output_via_attribute_access_is_accepted() -> None:
    registry = AgentRegistry()
    registry.register(AgentRegistration(name="general", description="General help."))
    registry.register(AgentRegistration(name="special", description="Special help."))

    class DecisionObj:
        agent = "special"
        confidence = 0.9
        reasoning = "Ok"

    classifier = FakeAgent(DecisionObj())
    special = FakeAgent({"answer": "from special"})
    agents: dict[str, FakeAgent] = {
        "special": special,
        "general": FakeAgent({"answer": "from general"}),
    }

    dispatcher = PydanticAIDispatcher(
        registry=registry,
        classifier_agent=classifier,
        get_agent=lambda name: agents.get(name),
        default_agent="general",
    )

    result = _run(dispatcher)
    assert result.agent_name == "special"


def test_dispatcher_classifier_output_missing_fields_via_attribute_access_raises() -> None:
    registry = AgentRegistry()
    registry.register(AgentRegistration(name="general", description="General help."))

    class DecisionObj:
        agent = "general"

    class BadClassifier:
        async def run(self, message: str, *, deps: Any) -> Any:  # noqa: ARG002
            return DecisionObj()

    agents: dict[str, Any] = {"general": FakeAgent({"answer": "from general"})}
    dispatcher = PydanticAIDispatcher(
        registry=registry,
        classifier_agent=BadClassifier(),
        get_agent=lambda name: agents.get(name),
        default_agent="general",
    )

    with pytest.raises(InvalidRouteDecision, match="must provide at least"):
        _run(dispatcher)


def test_dispatcher_pinned_agent_bypasses_classifier() -> None:
    registry = AgentRegistry()
    registry.register(AgentRegistration(name="general", description="General help."))
    registry.register(AgentRegistration(name="special", description="Special help."))

    classifier = FakeAgent(
        RouteDecision(agent="general", confidence=0.9, reasoning="Should not be called")
    )
    special = FakeAgent({"answer": "from special"})
    agents: dict[str, FakeAgent] = {
        "special": special,
        "general": FakeAgent({"answer": "from general"}),
    }

    dispatcher = PydanticAIDispatcher(
        registry=registry,
        classifier_agent=classifier,
        get_agent=lambda name: agents.get(name),
        default_agent="general",
    )

    # pinned bypass: should never call classifier
    result = _run(dispatcher, pinned_agent="special")
    assert classifier.called is False
    assert result.was_pinned is True
    assert result.agent_name == "special"
    assert result.output == {"answer": "from special"}


def test_dispatcher_classifier_routes_to_selected_agent() -> None:
    registry = AgentRegistry()
    registry.register(AgentRegistration(name="general", description="General help."))
    registry.register(AgentRegistration(name="special", description="Special help."))

    classifier = FakeAgent(RouteDecision(agent="special", confidence=0.8, reasoning="Match."))
    special = FakeAgent({"answer": "from special"})
    agents: dict[str, FakeAgent] = {
        "special": special,
        "general": FakeAgent({"answer": "from general"}),
    }

    dispatcher = PydanticAIDispatcher(
        registry=registry,
        classifier_agent=classifier,
        get_agent=lambda name: agents.get(name),
        default_agent="general",
    )

    result = _run(dispatcher)
    assert classifier.called is True
    assert result.was_pinned is False
    assert result.agent_name == "special"
    assert result.validated_decision is not None
    assert result.validated_decision.did_fallback is False


def test_dispatcher_pinned_agent_rejects_whitespace() -> None:
    registry = AgentRegistry()
    registry.register(AgentRegistration(name="general", description="General help."))
    classifier = FakeAgent(RouteDecision(agent="general", confidence=0.9, reasoning="Ok"))
    agents: dict[str, FakeAgent] = {"general": FakeAgent({"answer": "from general"})}

    dispatcher = PydanticAIDispatcher(
        registry=registry,
        classifier_agent=classifier,
        get_agent=lambda name: agents.get(name),
        default_agent="general",
    )

    with pytest.raises(InvalidRouteDecision, match="Pinned agent cannot be empty"):
        _run(dispatcher, pinned_agent="   ")
    assert classifier.called is False


def test_dispatcher_falls_back_when_classifier_selects_unknown_agent() -> None:
    registry = AgentRegistry()
    registry.register(AgentRegistration(name="general", description="General help."))
    registry.register(AgentRegistration(name="special", description="Special help."))

    classifier = FakeAgent(RouteDecision(agent="unknown", confidence=0.8, reasoning="Oops."))
    general = FakeAgent({"answer": "from general"})
    agents: dict[str, FakeAgent] = {
        "general": general,
        "special": FakeAgent({"answer": "from special"}),
    }

    dispatcher = PydanticAIDispatcher(
        registry=registry,
        classifier_agent=classifier,
        get_agent=lambda name: agents.get(name),
        default_agent="general",
    )

    with pytest.raises(InvalidRouteDecision):
        _run(dispatcher)


def test_dispatcher_agent_resolve_failure_emits_and_raises() -> None:
    registry = AgentRegistry()
    registry.register(AgentRegistration(name="general", description="General help."))
    registry.register(AgentRegistration(name="special", description="Special help."))

    classifier = FakeAgent(RouteDecision(agent="special", confidence=0.9, reasoning="Ok"))
    agents: dict[str, FakeAgent] = {"general": FakeAgent({"answer": "from general"})}
    events: list[RoutingEvent] = []

    dispatcher = PydanticAIDispatcher(
        registry=registry,
        classifier_agent=classifier,
        get_agent=lambda name: agents.get(name),
        default_agent="general",
        on_event=lambda e: events.append(e),
    )

    with pytest.raises(AgentNotFound):
        _run(dispatcher)

    kinds = [e.kind for e in events]
    assert "agent_resolve_failed" in kinds


def test_dispatcher_pinned_invalid_emits_then_routes() -> None:
    registry = AgentRegistry()
    registry.register(AgentRegistration(name="general", description="General help."))
    registry.register(AgentRegistration(name="special", description="Special help."))

    classifier = FakeAgent(RouteDecision(agent="special", confidence=0.9, reasoning="Ok"))
    agents: dict[str, FakeAgent] = {
        "special": FakeAgent({"answer": "from special"}),
        "general": FakeAgent({"answer": "from general"}),
    }
    events: list[RoutingEvent] = []

    dispatcher = PydanticAIDispatcher(
        registry=registry,
        classifier_agent=classifier,
        get_agent=lambda name: agents.get(name),
        default_agent="general",
        on_event=lambda e: events.append(e),
    )

    result = _run(dispatcher, pinned_agent="not-real")
    assert result.agent_name == "special"  # classifier route
    kinds = [e.kind for e in events]
    assert "pinned_invalid" in kinds
    assert "classifier_run_success" in kinds


def test_dispatcher_classifier_output_missing_fields_raises() -> None:
    registry = AgentRegistry()
    registry.register(AgentRegistration(name="general", description="General help."))

    class BadClassifier:
        async def run(self, message: str, *, deps: Any) -> Any:  # type: ignore[override]
            return {"agent": None}  # missing confidence

    agents: dict[str, FakeAgent] = {"general": FakeAgent({"answer": "from general"})}
    dispatcher = PydanticAIDispatcher(
        registry=registry,
        classifier_agent=BadClassifier(),
        get_agent=lambda name: agents.get(name),
        default_agent="general",
    )

    with pytest.raises(InvalidRouteDecision):
        _run(dispatcher)


def test_dispatcher_event_hook_failure_is_swallowed() -> None:
    registry = AgentRegistry()
    registry.register(AgentRegistration(name="general", description="General help."))

    classifier = FakeAgent(RouteDecision(agent="general", confidence=0.9, reasoning="Ok"))
    agents: dict[str, FakeAgent] = {"general": FakeAgent({"answer": "from general"})}

    def bad_hook(event: RoutingEvent) -> None:
        raise RuntimeError("boom")

    dispatcher = PydanticAIDispatcher(
        registry=registry,
        classifier_agent=classifier,
        get_agent=lambda name: agents.get(name),
        default_agent="general",
        on_event=bad_hook,
    )

    result = _run(dispatcher)
    assert result.agent_name == "general"


def _run(dispatcher: PydanticAIDispatcher, pinned_agent: str | None = None) -> DispatchResult:
    # tiny sync harness around an async method, without adding async test deps
    import asyncio

    async def go() -> DispatchResult:
        return await dispatcher.route_and_run(
            "hello",
            classifier_deps={"k": "v"},
            deps_for_agent=lambda _name: {"deps": True},
            pinned_agent=pinned_agent,
        )

    return asyncio.run(go())


def test_dispatcher_emits_events() -> None:
    registry = AgentRegistry()
    registry.register(AgentRegistration(name="general", description="General help."))
    classifier = FakeAgent(RouteDecision(agent="general", confidence=0.9, reasoning="Ok"))
    general = FakeAgent({"answer": "from general"})
    agents: dict[str, FakeAgent] = {"general": general}

    events: list[RoutingEvent] = []

    dispatcher = PydanticAIDispatcher(
        registry=registry,
        classifier_agent=classifier,
        get_agent=lambda name: agents.get(name),
        default_agent="general",
        on_event=lambda e: events.append(e),
    )

    _run(dispatcher)

    kinds = [e.kind for e in events]
    assert "classifier_run_start" in kinds
    assert "classifier_run_success" in kinds
    assert "decision_validated" in kinds
    assert "agent_resolve_success" in kinds
    assert "agent_run_success" in kinds


def test_dispatcher_route_and_stream_happy_path_yields_chunks_and_events() -> None:
    registry = AgentRegistry()
    registry.register(AgentRegistration(name="general", description="General help."))
    registry.register(AgentRegistration(name="special", description="Special help."))

    classifier = FakeAgent(RouteDecision(agent="special", confidence=0.8, reasoning="Ok"))
    special = FakeStreamingAgent(["a", "b", "c"])
    agents: dict[str, Any] = {"special": special}
    events: list[RoutingEvent] = []

    dispatcher = PydanticAIDispatcher(
        registry=registry,
        classifier_agent=classifier,
        get_agent=lambda name: agents.get(name),
        default_agent="general",
        on_event=lambda e: events.append(e),
    )

    chunks = _stream(dispatcher)
    assert [c.chunk for c in chunks] == ["a", "b", "c"]
    assert chunks[0].agent_name == "special"
    assert chunks[0].validated_decision is not None
    assert chunks[0].classifier_decision is not None
    assert chunks[1].validated_decision is None
    assert chunks[1].classifier_decision is None

    kinds = [e.kind for e in events]
    assert "classifier_run_start" in kinds
    assert "classifier_run_success" in kinds
    assert "decision_validated" in kinds
    assert "agent_resolve_success" in kinds
    assert kinds.count("agent_stream_chunk") == 3
    assert "agent_stream_end" in kinds
    assert "agent_run_success" in kinds


def test_dispatcher_route_and_stream_agent_resolve_failure_emits_and_raises() -> None:
    registry = AgentRegistry()
    registry.register(AgentRegistration(name="general", description="General help."))
    registry.register(AgentRegistration(name="special", description="Special help."))

    classifier = FakeAgent(RouteDecision(agent="special", confidence=0.9, reasoning="Ok"))
    agents: dict[str, Any] = {"general": FakeStreamingAgent(["x"])}
    events: list[RoutingEvent] = []

    dispatcher = PydanticAIDispatcher(
        registry=registry,
        classifier_agent=classifier,
        get_agent=lambda name: agents.get(name),
        default_agent="general",
        on_event=lambda e: events.append(e),
    )

    import asyncio

    async def go() -> None:
        async for _chunk in dispatcher.route_and_stream(
            "hello",
            classifier_deps={"k": "v"},
            deps_for_agent=lambda _name: {"deps": True},
        ):
            raise AssertionError("should not yield when agent resolution fails")

    with pytest.raises(AgentNotFound):
        asyncio.run(go())

    kinds = [e.kind for e in events]
    assert "agent_resolve_failed" in kinds


def test_dispatcher_route_and_stream_pinned_bypasses_classifier() -> None:
    registry = AgentRegistry()
    registry.register(AgentRegistration(name="general", description="General help."))
    registry.register(AgentRegistration(name="special", description="Special help."))

    classifier = FakeAgent(
        RouteDecision(agent="general", confidence=0.1, reasoning="Should not be called")
    )
    special = FakeStreamingAgent(["hi"])
    agents: dict[str, Any] = {"special": special}
    events: list[RoutingEvent] = []

    dispatcher = PydanticAIDispatcher(
        registry=registry,
        classifier_agent=classifier,
        get_agent=lambda name: agents.get(name),
        default_agent="general",
        on_event=lambda e: events.append(e),
    )

    import asyncio

    async def go() -> list[AgentStreamChunk]:
        out: list[AgentStreamChunk] = []
        async for chunk in dispatcher.route_and_stream(
            "hello",
            classifier_deps={"k": "v"},
            deps_for_agent=lambda _name: {"deps": True},
            pinned_agent="special",
        ):
            out.append(chunk)
        return out

    chunks = asyncio.run(go())
    assert classifier.called is False
    assert [c.chunk for c in chunks] == ["hi"]
    assert chunks[0].was_pinned is True

    kinds = [e.kind for e in events]
    assert "pinned_bypass" in kinds
    assert "agent_stream_end" in kinds
    assert "agent_run_success" in kinds


def test_dispatcher_route_and_stream_rejects_whitespace_pinned() -> None:
    registry = AgentRegistry()
    registry.register(AgentRegistration(name="general", description="General help."))
    registry.register(AgentRegistration(name="special", description="Special help."))

    classifier = FakeAgent(RouteDecision(agent="general", confidence=0.1, reasoning="Ok"))
    agents: dict[str, Any] = {"special": FakeStreamingAgent(["hi"])}

    dispatcher = PydanticAIDispatcher(
        registry=registry,
        classifier_agent=classifier,
        get_agent=lambda name: agents.get(name),
        default_agent="general",
    )

    import asyncio

    async def go() -> None:
        async for _chunk in dispatcher.route_and_stream(
            "hello",
            classifier_deps={"k": "v"},
            deps_for_agent=lambda _name: {"deps": True},
            pinned_agent=" ",
        ):
            raise AssertionError("should not stream when pinned agent is whitespace")

    with pytest.raises(InvalidRouteDecision, match="Pinned agent cannot be empty"):
        asyncio.run(go())
    assert classifier.called is False


def test_dispatcher_route_and_stream_pinned_invalid_emits_then_routes() -> None:
    registry = AgentRegistry()
    registry.register(AgentRegistration(name="general", description="General help."))
    registry.register(AgentRegistration(name="special", description="Special help."))

    classifier = FakeAgent(RouteDecision(agent="special", confidence=0.9, reasoning="Ok"))
    agents: dict[str, Any] = {"special": FakeStreamingAgent(["x"])}
    events: list[RoutingEvent] = []

    dispatcher = PydanticAIDispatcher(
        registry=registry,
        classifier_agent=classifier,
        get_agent=lambda name: agents.get(name),
        default_agent="general",
        on_event=lambda e: events.append(e),
    )

    import asyncio

    async def go() -> list[AgentStreamChunk]:
        out: list[AgentStreamChunk] = []
        async for chunk in dispatcher.route_and_stream(
            "hello",
            classifier_deps={"k": "v"},
            deps_for_agent=lambda _name: {"deps": True},
            pinned_agent="not-real",
        ):
            out.append(chunk)
        return out

    chunks = asyncio.run(go())
    assert [c.chunk for c in chunks] == ["x"]

    kinds = [e.kind for e in events]
    assert "pinned_invalid" in kinds
    assert "classifier_run_success" in kinds


def test_dispatcher_route_and_stream_pinned_raises_when_agent_not_streamable() -> None:
    registry = AgentRegistry()
    registry.register(AgentRegistration(name="general", description="General help."))
    registry.register(AgentRegistration(name="special", description="Special help."))

    classifier = FakeAgent(RouteDecision(agent="general", confidence=0.1, reasoning="Ok"))
    agents: dict[str, Any] = {"special": FakeAgent({"answer": "no stream"})}

    dispatcher = PydanticAIDispatcher(
        registry=registry,
        classifier_agent=classifier,
        get_agent=lambda name: agents.get(name),
        default_agent="general",
    )

    import asyncio

    async def go() -> None:
        async for _chunk in dispatcher.route_and_stream(
            "hello",
            classifier_deps={"k": "v"},
            deps_for_agent=lambda _name: {"deps": True},
            pinned_agent="special",
        ):
            raise AssertionError("should not yield when pinned agent isn't streamable")

    with pytest.raises(TypeError, match="missing run_stream"):
        asyncio.run(go())


def test_dispatcher_route_and_stream_pinned_raises_when_stream_result_missing_stream_text() -> None:
    registry = AgentRegistry()
    registry.register(AgentRegistration(name="general", description="General help."))
    registry.register(AgentRegistration(name="special", description="Special help."))

    classifier = FakeAgent(RouteDecision(agent="general", confidence=0.1, reasoning="Ok"))
    agents: dict[str, Any] = {"special": FakeStreamingAgentMissingStreamText()}

    dispatcher = PydanticAIDispatcher(
        registry=registry,
        classifier_agent=classifier,
        get_agent=lambda name: agents.get(name),
        default_agent="general",
    )

    import asyncio

    async def go() -> None:
        async for _chunk in dispatcher.route_and_stream(
            "hello",
            classifier_deps={"k": "v"},
            deps_for_agent=lambda _name: {"deps": True},
            pinned_agent="special",
        ):
            raise AssertionError("should not yield when stream_text isn't supported")

    with pytest.raises(TypeError, match="missing stream_text"):
        asyncio.run(go())


def test_dispatcher_route_and_stream_pinned_supports_stream_text_without_delta_kwarg() -> None:
    registry = AgentRegistry()
    registry.register(AgentRegistration(name="general", description="General help."))
    registry.register(AgentRegistration(name="special", description="Special help."))

    classifier = FakeAgent(RouteDecision(agent="general", confidence=0.1, reasoning="Ok"))
    agents: dict[str, Any] = {"special": FakeStreamingAgentNoDelta(["a", "b"])}

    dispatcher = PydanticAIDispatcher(
        registry=registry,
        classifier_agent=classifier,
        get_agent=lambda name: agents.get(name),
        default_agent="general",
    )

    import asyncio

    async def go() -> list[AgentStreamChunk]:
        out: list[AgentStreamChunk] = []
        async for chunk in dispatcher.route_and_stream(
            "hello",
            classifier_deps={"k": "v"},
            deps_for_agent=lambda _name: {"deps": True},
            pinned_agent="special",
        ):
            out.append(chunk)
        return out

    chunks = asyncio.run(go())
    assert [c.chunk for c in chunks] == ["a", "b"]


def test_dispatcher_route_and_stream_raises_when_agent_not_streamable() -> None:
    registry = AgentRegistry()
    registry.register(AgentRegistration(name="general", description="General help."))
    registry.register(AgentRegistration(name="special", description="Special help."))

    classifier = FakeAgent(RouteDecision(agent="special", confidence=0.9, reasoning="Ok"))
    # This agent implements only `run`, not `run_stream` (a common misconfiguration).
    agents: dict[str, Any] = {"special": FakeAgent({"answer": "no stream"})}

    dispatcher = PydanticAIDispatcher(
        registry=registry,
        classifier_agent=classifier,
        get_agent=lambda name: agents.get(name),
        default_agent="general",
    )

    import asyncio

    async def go() -> None:
        async for _chunk in dispatcher.route_and_stream(
            "hello",
            classifier_deps={"k": "v"},
            deps_for_agent=lambda _name: {"deps": True},
        ):
            raise AssertionError("should not yield when streaming isn't supported")

    with pytest.raises(TypeError, match="missing run_stream"):
        asyncio.run(go())


def test_dispatcher_route_and_stream_raises_when_stream_result_missing_stream_text() -> None:
    registry = AgentRegistry()
    registry.register(AgentRegistration(name="general", description="General help."))
    registry.register(AgentRegistration(name="special", description="Special help."))

    classifier = FakeAgent(RouteDecision(agent="special", confidence=0.9, reasoning="Ok"))
    agents: dict[str, Any] = {"special": FakeStreamingAgentMissingStreamText()}

    dispatcher = PydanticAIDispatcher(
        registry=registry,
        classifier_agent=classifier,
        get_agent=lambda name: agents.get(name),
        default_agent="general",
    )

    import asyncio

    async def go() -> None:
        async for _chunk in dispatcher.route_and_stream(
            "hello",
            classifier_deps={"k": "v"},
            deps_for_agent=lambda _name: {"deps": True},
        ):
            raise AssertionError("should not yield when stream_text isn't supported")

    with pytest.raises(TypeError, match="missing stream_text"):
        asyncio.run(go())


def test_dispatcher_route_and_stream_supports_stream_text_without_delta_kwarg() -> None:
    registry = AgentRegistry()
    registry.register(AgentRegistration(name="general", description="General help."))
    registry.register(AgentRegistration(name="special", description="Special help."))

    classifier = FakeAgent(RouteDecision(agent="special", confidence=0.9, reasoning="Ok"))
    agents: dict[str, Any] = {"special": FakeStreamingAgentNoDelta(["a", "b"])}

    dispatcher = PydanticAIDispatcher(
        registry=registry,
        classifier_agent=classifier,
        get_agent=lambda name: agents.get(name),
        default_agent="general",
    )

    chunks = _stream(dispatcher)
    assert [c.chunk for c in chunks] == ["a", "b"]


def test_dispatcher_route_and_stream_streaming_classifier_is_consumed_internally() -> None:
    registry = AgentRegistry()
    registry.register(AgentRegistration(name="general", description="General help."))
    registry.register(AgentRegistration(name="special", description="Special help."))

    classifier = FakeStreamingClassifier(
        RouteDecision(agent="special", confidence=0.9, reasoning="Ok")
    )
    agents: dict[str, Any] = {"special": FakeStreamingAgent(["x"])}
    events: list[RoutingEvent] = []

    dispatcher = PydanticAIDispatcher(
        registry=registry,
        classifier_agent=classifier,
        get_agent=lambda name: agents.get(name),
        default_agent="general",
        on_event=lambda e: events.append(e),
    )

    import asyncio

    async def go() -> list[AgentStreamChunk]:
        out: list[AgentStreamChunk] = []
        async for chunk in dispatcher.route_and_stream(
            "hello",
            classifier_deps={"k": "v"},
            deps_for_agent=lambda _name: {"deps": True},
            stream_classifier=True,
        ):
            out.append(chunk)
        return out

    chunks = asyncio.run(go())
    assert [c.chunk for c in chunks] == ["x"]
    kinds = [e.kind for e in events]
    assert "classifier_run_start" in kinds
    assert "classifier_run_success" in kinds


def test_dispatcher_route_and_stream_streaming_classifier_without_final_event_raises() -> None:
    registry = AgentRegistry()
    registry.register(AgentRegistration(name="general", description="General help."))
    registry.register(AgentRegistration(name="special", description="Special help."))

    classifier = FakeStreamingClassifier(
        RouteDecision(agent="special", confidence=0.9, reasoning="Ok"),
        include_final_event=False,
    )
    agents: dict[str, Any] = {"special": FakeStreamingAgent(["x"])}

    dispatcher = PydanticAIDispatcher(
        registry=registry,
        classifier_agent=classifier,
        get_agent=lambda name: agents.get(name),
        default_agent="general",
    )

    import asyncio

    async def go() -> None:
        async for _chunk in dispatcher.route_and_stream(
            "hello",
            classifier_deps={"k": "v"},
            deps_for_agent=lambda _name: {"deps": True},
            stream_classifier=True,
        ):
            raise AssertionError(
                "should not stream if classifier fails to produce a final decision"
            )

    with pytest.raises(InvalidRouteDecision, match="no final result event"):
        asyncio.run(go())


def test_dispatcher_route_and_stream_streaming_classifier_falls_back_to_run() -> None:
    registry = AgentRegistry()
    registry.register(AgentRegistration(name="general", description="General help."))
    registry.register(AgentRegistration(name="special", description="Special help."))

    classifier = FakeAgent(RouteDecision(agent="special", confidence=0.9, reasoning="Ok"))
    agents: dict[str, Any] = {"special": FakeStreamingAgent(["x"])}

    dispatcher = PydanticAIDispatcher(
        registry=registry,
        classifier_agent=classifier,
        get_agent=lambda name: agents.get(name),
        default_agent="general",
    )

    import asyncio

    async def go() -> list[AgentStreamChunk]:
        out: list[AgentStreamChunk] = []
        async for chunk in dispatcher.route_and_stream(
            "hello",
            classifier_deps={"k": "v"},
            deps_for_agent=lambda _name: {"deps": True},
            stream_classifier=True,
        ):
            out.append(chunk)
        return out

    chunks = asyncio.run(go())
    assert classifier.called is True
    assert [c.chunk for c in chunks] == ["x"]


def test_dispatcher_route_and_stream_streaming_classifier_run_stream_stream_output_sync_iter() -> (
    None
):
    registry = AgentRegistry()
    registry.register(AgentRegistration(name="general", description="General help."))
    registry.register(AgentRegistration(name="special", description="Special help."))

    streamed = FakeClassifierStreamedRunResultOutputSync([{"agent": "special", "confidence": 0.9}])
    classifier = FakeStreamingClassifierRunStream(streamed)
    agents: dict[str, Any] = {"special": FakeStreamingAgent(["x"])}

    dispatcher = PydanticAIDispatcher(
        registry=registry,
        classifier_agent=classifier,
        get_agent=lambda name: agents.get(name),
        default_agent="general",
    )

    import asyncio

    async def go() -> list[AgentStreamChunk]:
        out: list[AgentStreamChunk] = []
        async for chunk in dispatcher.route_and_stream(
            "hello",
            classifier_deps={"k": "v"},
            deps_for_agent=lambda _name: {"deps": True},
            stream_classifier=True,
        ):
            out.append(chunk)
        return out

    chunks = asyncio.run(go())
    assert [c.chunk for c in chunks] == ["x"]


def test_dispatcher_route_and_stream_streaming_classifier_run_stream_stream_output_async_iter() -> (
    None
):
    registry = AgentRegistry()
    registry.register(AgentRegistration(name="general", description="General help."))
    registry.register(AgentRegistration(name="special", description="Special help."))

    streamed = FakeClassifierStreamedRunResultOutputAsync([{"agent": "special", "confidence": 0.9}])
    classifier = FakeStreamingClassifierRunStream(streamed)
    agents: dict[str, Any] = {"special": FakeStreamingAgent(["x"])}

    dispatcher = PydanticAIDispatcher(
        registry=registry,
        classifier_agent=classifier,
        get_agent=lambda name: agents.get(name),
        default_agent="general",
    )

    import asyncio

    async def go() -> list[AgentStreamChunk]:
        out: list[AgentStreamChunk] = []
        async for chunk in dispatcher.route_and_stream(
            "hello",
            classifier_deps={"k": "v"},
            deps_for_agent=lambda _name: {"deps": True},
            stream_classifier=True,
        ):
            out.append(chunk)
        return out

    chunks = asyncio.run(go())
    assert [c.chunk for c in chunks] == ["x"]


def test_dispatcher_route_and_stream_streaming_classifier_run_stream_stream_text_no_delta() -> None:
    registry = AgentRegistry()
    registry.register(AgentRegistration(name="general", description="General help."))
    registry.register(AgentRegistration(name="special", description="Special help."))

    streamed = FakeClassifierStreamedRunResultTextNoDelta([{"agent": "special", "confidence": 0.9}])
    classifier = FakeStreamingClassifierRunStream(streamed)
    agents: dict[str, Any] = {"special": FakeStreamingAgent(["x"])}

    dispatcher = PydanticAIDispatcher(
        registry=registry,
        classifier_agent=classifier,
        get_agent=lambda name: agents.get(name),
        default_agent="general",
    )

    import asyncio

    async def go() -> list[AgentStreamChunk]:
        out: list[AgentStreamChunk] = []
        async for chunk in dispatcher.route_and_stream(
            "hello",
            classifier_deps={"k": "v"},
            deps_for_agent=lambda _name: {"deps": True},
            stream_classifier=True,
        ):
            out.append(chunk)
        return out

    chunks = asyncio.run(go())
    assert [c.chunk for c in chunks] == ["x"]


def test_dispatcher_route_and_stream_streaming_classifier_run_stream_no_stream_methods_raises() -> (
    None
):
    registry = AgentRegistry()
    registry.register(AgentRegistration(name="general", description="General help."))
    registry.register(AgentRegistration(name="special", description="Special help."))

    classifier = FakeStreamingClassifierRunStream(FakeClassifierStreamedRunResultNoStreams())
    agents: dict[str, Any] = {"special": FakeStreamingAgent(["x"])}
    dispatcher = PydanticAIDispatcher(
        registry=registry,
        classifier_agent=classifier,
        get_agent=lambda name: agents.get(name),
        default_agent="general",
    )

    import asyncio

    async def go() -> None:
        async for _chunk in dispatcher.route_and_stream(
            "hello",
            classifier_deps={"k": "v"},
            deps_for_agent=lambda _name: {"deps": True},
            stream_classifier=True,
        ):
            raise AssertionError("should not yield if classifier stream is unusable")

    with pytest.raises(InvalidRouteDecision, match="does not provide stream_output"):
        asyncio.run(go())


def test_dispatcher_route_and_stream_streaming_classifier_run_stream_no_output_raises() -> None:
    registry = AgentRegistry()
    registry.register(AgentRegistration(name="general", description="General help."))
    registry.register(AgentRegistration(name="special", description="Special help."))

    classifier = FakeStreamingClassifierRunStream(FakeClassifierStreamedRunResultNoOutput())
    agents: dict[str, Any] = {"special": FakeStreamingAgent(["x"])}
    dispatcher = PydanticAIDispatcher(
        registry=registry,
        classifier_agent=classifier,
        get_agent=lambda name: agents.get(name),
        default_agent="general",
    )

    import asyncio

    async def go() -> None:
        async for _chunk in dispatcher.route_and_stream(
            "hello",
            classifier_deps={"k": "v"},
            deps_for_agent=lambda _name: {"deps": True},
            stream_classifier=True,
        ):
            raise AssertionError("should not yield if classifier produced no output")

    with pytest.raises(InvalidRouteDecision, match="produced no output"):
        asyncio.run(go())


def test_stream_responses_happy_path_exposes_streamed_run() -> None:
    registry = AgentRegistry()
    registry.register(AgentRegistration(name="general", description="General help."))
    registry.register(AgentRegistration(name="special", description="Special help."))

    classifier = FakeAgent(RouteDecision(agent="special", confidence=0.8, reasoning="Ok"))
    special = FakeResponseStreamingAgent([({"partial": 1}, False), ({"final": True}, True)])
    agents: dict[str, Any] = {"special": special}
    events: list[RoutingEvent] = []

    dispatcher = PydanticAIDispatcher(
        registry=registry,
        classifier_agent=classifier,
        get_agent=lambda name: agents.get(name),
        default_agent="general",
        on_event=lambda e: events.append(e),
    )

    import asyncio

    async def go() -> list[AgentResponseStreamItem]:
        out: list[AgentResponseStreamItem] = []
        async with dispatcher.route_and_stream_responses(
            "hello",
            classifier_deps={"k": "v"},
            deps_for_agent=lambda _name: {"deps": True},
        ) as session:
            assert session.agent_name == "special"
            assert session.streamed_run is not None
            async for item in session.iter_responses():
                out.append(item)
        return out

    items = asyncio.run(go())
    assert [i.model_response for i in items] == [{"partial": 1}, {"final": True}]
    assert [i.is_last for i in items] == [False, True]
    assert items[0].validated_decision is not None
    assert items[0].classifier_decision is not None
    assert items[1].validated_decision is None
    assert items[1].classifier_decision is None
    assert items[0].was_pinned is False

    kinds = [e.kind for e in events]
    assert "classifier_run_start" in kinds
    assert "classifier_run_success" in kinds
    assert "decision_validated" in kinds
    assert "agent_resolve_success" in kinds
    assert kinds.count("agent_stream_response") == 2
    assert "agent_stream_end" in kinds
    assert "agent_run_success" in kinds


def test_dispatcher_route_and_stream_responses_pinned_bypasses_classifier() -> None:
    registry = AgentRegistry()
    registry.register(AgentRegistration(name="general", description="General help."))
    registry.register(AgentRegistration(name="special", description="Special help."))

    classifier = FakeAgent(
        RouteDecision(agent="general", confidence=0.1, reasoning="Should not be called")
    )
    special = FakeResponseStreamingAgent([("x", True)])
    agents: dict[str, Any] = {"special": special}
    events: list[RoutingEvent] = []

    dispatcher = PydanticAIDispatcher(
        registry=registry,
        classifier_agent=classifier,
        get_agent=lambda name: agents.get(name),
        default_agent="general",
        on_event=lambda e: events.append(e),
    )

    import asyncio

    async def go() -> list[AgentResponseStreamItem]:
        out: list[AgentResponseStreamItem] = []
        async with dispatcher.route_and_stream_responses(
            "hello",
            classifier_deps={"k": "v"},
            deps_for_agent=lambda _name: {"deps": True},
            pinned_agent="special",
        ) as session:
            async for item in session.iter_responses():
                out.append(item)
        return out

    items = asyncio.run(go())
    assert classifier.called is False
    assert [i.model_response for i in items] == ["x"]
    assert items[0].was_pinned is True
    assert items[0].validated_decision is None
    assert items[0].classifier_decision is None

    kinds = [e.kind for e in events]
    assert "pinned_bypass" in kinds
    assert "classifier_run_start" not in kinds


def test_route_and_stream_responses_requires_final_streaming_decision() -> None:
    registry = AgentRegistry()
    registry.register(AgentRegistration(name="general", description="General help."))
    registry.register(AgentRegistration(name="special", description="Special help."))

    classifier = FakeStreamingClassifier(
        RouteDecision(agent="special", confidence=0.9, reasoning="Ok"),
        include_final_event=False,
    )
    agents: dict[str, Any] = {"special": FakeResponseStreamingAgent([("x", True)])}

    dispatcher = PydanticAIDispatcher(
        registry=registry,
        classifier_agent=classifier,
        get_agent=lambda name: agents.get(name),
        default_agent="general",
    )

    import asyncio

    async def go() -> None:
        async with dispatcher.route_and_stream_responses(
            "hello",
            classifier_deps={"k": "v"},
            deps_for_agent=lambda _name: {"deps": True},
            stream_classifier=True,
        ):
            raise AssertionError("should not stream when classifier omits final decision")

    with pytest.raises(InvalidRouteDecision, match="no final result event"):
        asyncio.run(go())


def test_dispatcher_route_and_stream_responses_pinned_invalid_emits_then_routes() -> None:
    registry = AgentRegistry()
    registry.register(AgentRegistration(name="general", description="General help."))
    registry.register(AgentRegistration(name="special", description="Special help."))

    classifier = FakeAgent(RouteDecision(agent="special", confidence=0.9, reasoning="Ok"))
    agents: dict[str, Any] = {"special": FakeResponseStreamingAgent([("x", True)])}
    events: list[RoutingEvent] = []

    dispatcher = PydanticAIDispatcher(
        registry=registry,
        classifier_agent=classifier,
        get_agent=lambda name: agents.get(name),
        default_agent="general",
        on_event=lambda e: events.append(e),
    )

    import asyncio

    async def go() -> list[AgentResponseStreamItem]:
        out: list[AgentResponseStreamItem] = []
        async with dispatcher.route_and_stream_responses(
            "hello",
            classifier_deps={"k": "v"},
            deps_for_agent=lambda _name: {"deps": True},
            pinned_agent="not-real",
        ) as session:
            async for item in session.iter_responses():
                out.append(item)
        return out

    items = asyncio.run(go())
    assert [i.model_response for i in items] == ["x"]

    kinds = [e.kind for e in events]
    assert "pinned_invalid" in kinds
    assert "classifier_run_success" in kinds


def test_stream_responses_raises_when_missing_stream_responses() -> None:
    registry = AgentRegistry()
    registry.register(AgentRegistration(name="general", description="General help."))
    registry.register(AgentRegistration(name="special", description="Special help."))

    classifier = FakeAgent(RouteDecision(agent="special", confidence=0.9, reasoning="Ok"))
    agents: dict[str, Any] = {"special": FakeStreamingAgentMissingStreamResponses()}

    dispatcher = PydanticAIDispatcher(
        registry=registry,
        classifier_agent=classifier,
        get_agent=lambda name: agents.get(name),
        default_agent="general",
    )

    import asyncio

    async def go() -> None:
        async with dispatcher.route_and_stream_responses(
            "hello",
            classifier_deps={"k": "v"},
            deps_for_agent=lambda _name: {"deps": True},
        ) as session:
            async for _item in session.iter_responses():
                raise AssertionError("should not yield when stream_responses isn't supported")

    with pytest.raises(TypeError, match="missing stream_responses"):
        asyncio.run(go())


def test_dispatcher_route_and_stream_responses_raises_when_agent_not_streamable() -> None:
    registry = AgentRegistry()
    registry.register(AgentRegistration(name="general", description="General help."))
    registry.register(AgentRegistration(name="special", description="Special help."))

    classifier = FakeAgent(RouteDecision(agent="special", confidence=0.9, reasoning="Ok"))
    agents: dict[str, Any] = {"special": FakeAgent({"answer": "no stream"})}

    dispatcher = PydanticAIDispatcher(
        registry=registry,
        classifier_agent=classifier,
        get_agent=lambda name: agents.get(name),
        default_agent="general",
    )

    import asyncio

    async def go() -> None:
        async with dispatcher.route_and_stream_responses(
            "hello",
            classifier_deps={"k": "v"},
            deps_for_agent=lambda _name: {"deps": True},
        ):
            raise AssertionError("should not enter when agent isn't streamable")

    with pytest.raises(TypeError, match="missing run_stream"):
        asyncio.run(go())


def _stream(dispatcher: PydanticAIDispatcher) -> list[AgentStreamChunk]:
    import asyncio

    async def go() -> list[AgentStreamChunk]:
        out: list[AgentStreamChunk] = []
        async for chunk in dispatcher.route_and_stream(
            "hello",
            classifier_deps={"k": "v"},
            deps_for_agent=lambda _name: {"deps": True},
        ):
            out.append(chunk)
        return out

    return asyncio.run(go())
