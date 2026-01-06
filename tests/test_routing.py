from agent_registry_router.core import (
    AgentRegistration,
    AgentRegistry,
    RouteDecision,
    validate_route_decision,
)


def test_validate_route_decision_accepts_routable_agent() -> None:
    registry = AgentRegistry()
    registry.register(AgentRegistration(name="general", description="General help."))

    decision = RouteDecision(agent="GENERAL", confidence=0.9, reasoning="Clear match.")
    validated = validate_route_decision(decision, registry=registry, default_agent="general")

    assert validated.agent == "general"
    assert validated.did_fallback is False
    assert validated.fallback_reason is None
    assert validated.confidence == 0.9


def test_validate_route_decision_falls_back_to_default_agent_when_invalid() -> None:
    registry = AgentRegistry()
    registry.register(AgentRegistration(name="general", description="General help."))
    registry.register(AgentRegistration(name="special", description="Specialized help."))

    decision = RouteDecision(agent="does_not_exist", confidence=0.8, reasoning="Guess.")
    validated = validate_route_decision(decision, registry=registry, default_agent="general")

    assert validated.did_fallback is True
    assert validated.agent == "general"
    assert validated.confidence == 0.5
    assert validated.fallback_reason is not None


def test_validate_route_decision_falls_back_to_first_routable_if_default_not_registered() -> None:
    registry = AgentRegistry()
    registry.register(AgentRegistration(name="special", description="Specialized help."))

    decision = RouteDecision(agent="does_not_exist", confidence=0.2, reasoning="Guess.")
    validated = validate_route_decision(decision, registry=registry, default_agent="general")

    assert validated.did_fallback is True
    assert validated.agent == "special"
    assert validated.fallback_reason is not None


