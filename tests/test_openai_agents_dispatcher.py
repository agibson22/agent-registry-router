from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import pytest

from agent_registry_router.adapters.openai_agents import (
    DispatchResult,
    OpenAIAgentsDispatcher,
    StreamEvent,
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
    final_output: Any


@dataclass
class FakeStreamEvent:
    type: str
    data: Any = None


class FakeRunResultStreaming:
    def __init__(self, events: list[FakeStreamEvent], final_output: Any = None) -> None:
        self._events = events
        self.final_output = final_output

    async def stream_events(self) -> AsyncIterator[FakeStreamEvent]:
        for event in self._events:
            yield event


class FakeAgent:
    def __init__(self, name: str) -> None:
        self.name = name


class FakeRunner:
    def __init__(
        self,
        run_results: dict[str, Any] | None = None,
        stream_results: dict[str, FakeRunResultStreaming] | None = None,
    ) -> None:
        self._run_results = run_results or {}
        self._stream_results = stream_results or {}
        self.run_calls: list[tuple[str, str]] = []

    async def run(
        self, agent: Any, input: str, **kwargs: Any  # noqa: A002
    ) -> FakeRunResult:
        self.run_calls.append((agent.name, input))
        output = self._run_results.get(agent.name, "default output")
        return FakeRunResult(final_output=output)

    def run_streamed(
        self, agent: Any, input: str, **kwargs: Any  # noqa: A002
    ) -> FakeRunResultStreaming:
        return self._stream_results.get(
            agent.name,
            FakeRunResultStreaming([], final_output="streamed output"),
        )


def _make_registry() -> AgentRegistry:
    registry = AgentRegistry()
    registry.register(AgentRegistration(name="billing", description="Handles billing."))
    registry.register(
        AgentRegistration(name="technical", description="Handles technical issues.")
    )
    registry.register(
        AgentRegistration(name="general", description="Handles general inquiries.")
    )
    return registry


def _make_agents() -> dict[str, FakeAgent]:
    return {
        "billing": FakeAgent("billing"),
        "technical": FakeAgent("technical"),
        "general": FakeAgent("general"),
    }


def _make_dispatcher(
    runner: FakeRunner | None = None,
    classifier_output: Any = None,
    on_event: Any = None,
) -> OpenAIAgentsDispatcher:
    agents = _make_agents()
    classifier = FakeAgent("classifier")

    if classifier_output is None:
        classifier_output = {"agent": "billing", "confidence": 0.9, "reasoning": "test"}

    if runner is None:
        runner = FakeRunner(run_results={"classifier": classifier_output})

    return OpenAIAgentsDispatcher(
        registry=_make_registry(),
        classifier_agent=classifier,
        runner=runner,
        get_agent=lambda name: agents.get(name),
        default_agent="general",
        on_event=on_event,
    )


@pytest.mark.anyio
async def test_route_and_run_basic() -> None:
    runner = FakeRunner(
        run_results={
            "classifier": {"agent": "billing", "confidence": 0.9, "reasoning": "test"},
            "billing": "Billing response",
        }
    )
    dispatcher = _make_dispatcher(runner=runner)
    result = await dispatcher.route_and_run("I was charged twice")

    assert isinstance(result, DispatchResult)
    assert result.agent_name == "billing"
    assert result.output == "Billing response"
    assert result.was_pinned is False
    assert result.classifier_decision is not None
    assert result.validated_decision is not None


@pytest.mark.anyio
async def test_route_and_run_pinned_bypass() -> None:
    runner = FakeRunner(run_results={"billing": "Pinned response"})
    dispatcher = _make_dispatcher(runner=runner)
    result = await dispatcher.route_and_run("anything", pinned_agent="billing")

    assert result.agent_name == "billing"
    assert result.output == "Pinned response"
    assert result.was_pinned is True
    assert result.classifier_decision is None
    assert result.validated_decision is None
    assert len(runner.run_calls) == 1
    assert runner.run_calls[0][0] == "billing"


@pytest.mark.anyio
async def test_route_and_run_pinned_invalid_falls_through() -> None:
    runner = FakeRunner(
        run_results={
            "classifier": {
                "agent": "general",
                "confidence": 0.8,
                "reasoning": "fallback",
            },
            "general": "General response",
        }
    )
    dispatcher = _make_dispatcher(runner=runner)
    result = await dispatcher.route_and_run("test", pinned_agent="nonexistent")

    assert result.agent_name == "general"
    assert result.was_pinned is False


@pytest.mark.anyio
async def test_route_and_run_agent_not_found() -> None:
    runner = FakeRunner(
        run_results={
            "classifier": {"agent": "billing", "confidence": 0.9, "reasoning": "test"},
        }
    )
    dispatcher = OpenAIAgentsDispatcher(
        registry=_make_registry(),
        classifier_agent=FakeAgent("classifier"),
        runner=runner,
        get_agent=lambda name: None,
        default_agent="general",
    )

    with pytest.raises(AgentNotFound):
        await dispatcher.route_and_run("test")


@pytest.mark.anyio
async def test_route_and_run_invalid_classifier_output() -> None:
    runner = FakeRunner(run_results={"classifier": "just a string"})
    dispatcher = _make_dispatcher(runner=runner)

    with pytest.raises(InvalidRouteDecision):
        await dispatcher.route_and_run("test")


@pytest.mark.anyio
async def test_route_and_run_emits_events() -> None:
    events: list[RoutingEvent] = []
    runner = FakeRunner(
        run_results={
            "classifier": {"agent": "billing", "confidence": 0.9, "reasoning": "test"},
            "billing": "response",
        }
    )
    dispatcher = _make_dispatcher(runner=runner, on_event=events.append)
    await dispatcher.route_and_run("test")

    event_kinds = [e.kind for e in events]
    assert "classifier_run_start" in event_kinds
    assert "classifier_run_success" in event_kinds
    assert "decision_validated" in event_kinds
    assert "agent_resolve_success" in event_kinds
    assert "agent_run_success" in event_kinds


@pytest.mark.anyio
async def test_route_and_run_coerces_route_decision() -> None:
    decision = RouteDecision(agent="technical", confidence=0.85, reasoning="direct")
    runner = FakeRunner(
        run_results={"classifier": decision, "technical": "Tech response"}
    )
    dispatcher = _make_dispatcher(runner=runner)
    result = await dispatcher.route_and_run("error in app")

    assert result.agent_name == "technical"
    assert result.classifier_decision is not None
    assert result.classifier_decision.agent == "technical"


@pytest.mark.anyio
async def test_route_and_run_pinned_empty_raises() -> None:
    dispatcher = _make_dispatcher()
    with pytest.raises(InvalidRouteDecision):
        await dispatcher.route_and_run("test", pinned_agent="   ")


@pytest.mark.anyio
async def test_route_and_stream_basic() -> None:
    stream_events = [
        FakeStreamEvent(type="raw_response_event", data="chunk1"),
        FakeStreamEvent(type="raw_response_event", data="chunk2"),
    ]
    runner = FakeRunner(
        run_results={
            "classifier": {"agent": "billing", "confidence": 0.9, "reasoning": "test"},
        },
        stream_results={
            "billing": FakeRunResultStreaming(stream_events, final_output="done"),
        },
    )
    dispatcher = _make_dispatcher(runner=runner)
    collected: list[StreamEvent] = []
    async for event in dispatcher.route_and_stream("billing question"):
        collected.append(event)

    assert len(collected) == 2
    assert collected[0].agent_name == "billing"
    assert collected[0].validated_decision is not None
    assert collected[0].was_pinned is False
    assert collected[1].event_index == 1


@pytest.mark.anyio
async def test_route_and_stream_pinned() -> None:
    stream_events = [FakeStreamEvent(type="text", data="pinned chunk")]
    runner = FakeRunner(
        stream_results={
            "billing": FakeRunResultStreaming(stream_events),
        },
    )
    dispatcher = _make_dispatcher(runner=runner)
    collected: list[StreamEvent] = []
    async for event in dispatcher.route_and_stream("test", pinned_agent="billing"):
        collected.append(event)

    assert len(collected) == 1
    assert collected[0].was_pinned is True
    assert collected[0].agent_name == "billing"


@pytest.mark.anyio
async def test_event_hook_failure_does_not_break_routing() -> None:
    def bad_hook(event: RoutingEvent) -> None:
        raise ValueError("hook exploded")

    runner = FakeRunner(
        run_results={
            "classifier": {"agent": "billing", "confidence": 0.9, "reasoning": "test"},
            "billing": "still works",
        }
    )
    dispatcher = _make_dispatcher(runner=runner, on_event=bad_hook)
    result = await dispatcher.route_and_run("test")

    assert result.output == "still works"
