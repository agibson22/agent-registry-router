import pytest

from agent_registry_router.core import (
    AgentRegistration,
    AgentRegistry,
    InvalidFallback,
    InvalidRouteDecision,
    RegistryError,
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
    with pytest.raises(InvalidRouteDecision):
        validate_route_decision(decision, registry=registry, default_agent="general")


def test_validate_route_decision_falls_back_to_first_routable_if_default_not_registered() -> None:
    registry = AgentRegistry()
    registry.register(AgentRegistration(name="special", description="Specialized help."))

    decision = RouteDecision(agent="does_not_exist", confidence=0.2, reasoning="Guess.")
    with pytest.raises(InvalidFallback):
        validate_route_decision(decision, registry=registry, default_agent="general")


def test_validate_route_decision_raises_when_registry_empty() -> None:
    registry = AgentRegistry()
    decision = RouteDecision(agent="general", confidence=0.2, reasoning="Guess.")

    with pytest.raises(InvalidFallback):
        validate_route_decision(decision, registry=registry, default_agent="general")


def test_validate_route_decision_raises_when_default_not_routable() -> None:
    registry = AgentRegistry()
    registry.register(AgentRegistration(name="special", description="Specialized help."))

    decision = RouteDecision(agent="special", confidence=0.9, reasoning="Ok")
    with pytest.raises(InvalidFallback):
        validate_route_decision(decision, registry=registry, default_agent="general")


def test_registry_from_descriptions_and_lists() -> None:
    descriptions = {"general": "General help.", "special": "Special help."}
    registry = AgentRegistry.from_descriptions(descriptions)

    assert set(registry.all_names()) == {"general", "special"}
    assert set(registry.routable_names()) == {"general", "special"}
    assert registry.descriptions() == descriptions
    assert registry.routable_descriptions() == descriptions


def test_registry_rejects_empty_name() -> None:
    registry = AgentRegistry()
    with pytest.raises(RegistryError):
        registry.register(AgentRegistration(name="   ", description="bad"))


def test_registry_get_normalizes_name() -> None:
    registry = AgentRegistry()
    registry.register(AgentRegistration(name="general", description="General help."))

    reg = registry.get("  GENERAL  ")
    assert reg is not None
    assert reg.name == "general"
