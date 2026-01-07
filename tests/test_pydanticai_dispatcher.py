from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from agent_registry_router.adapters.pydantic_ai import PydanticAIDispatcher
import pytest

from agent_registry_router.core import (
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
        self.last_deps: Optional[Any] = None

    async def run(self, message: str, *, deps: Any) -> FakeRunResult:
        self.called = True
        self.last_deps = deps
        return FakeRunResult(self._output)


def test_dispatcher_pinned_agent_bypasses_classifier() -> None:
    registry = AgentRegistry()
    registry.register(AgentRegistration(name="general", description="General help."))
    registry.register(AgentRegistration(name="special", description="Special help."))

    classifier = FakeAgent(RouteDecision(agent="general", confidence=0.9, reasoning="Should not be called"))
    special = FakeAgent({"answer": "from special"})
    agents: Dict[str, FakeAgent] = {"special": special, "general": FakeAgent({"answer": "from general"})}

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
    agents: Dict[str, FakeAgent] = {"special": special, "general": FakeAgent({"answer": "from general"})}

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


def test_dispatcher_falls_back_when_classifier_selects_unknown_agent() -> None:
    registry = AgentRegistry()
    registry.register(AgentRegistration(name="general", description="General help."))
    registry.register(AgentRegistration(name="special", description="Special help."))

    classifier = FakeAgent(RouteDecision(agent="unknown", confidence=0.8, reasoning="Oops."))
    general = FakeAgent({"answer": "from general"})
    agents: Dict[str, FakeAgent] = {"general": general, "special": FakeAgent({"answer": "from special"})}

    dispatcher = PydanticAIDispatcher(
        registry=registry,
        classifier_agent=classifier,
        get_agent=lambda name: agents.get(name),
        default_agent="general",
    )

    with pytest.raises(InvalidRouteDecision):
        _run(dispatcher)


def _run(dispatcher: PydanticAIDispatcher, pinned_agent: Optional[str] = None):
    # tiny sync harness around an async method, without adding async test deps
    import asyncio

    async def go():
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
    agents: Dict[str, FakeAgent] = {"general": general}

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


