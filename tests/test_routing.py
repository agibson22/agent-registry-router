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


def test_validate_route_decision_allow_fallback_swaps_to_default() -> None:
    registry = AgentRegistry()
    registry.register(AgentRegistration(name="general", description="General help."))
    registry.register(AgentRegistration(name="special", description="Specialized help."))

    decision = RouteDecision(agent="does_not_exist", confidence=0.8, reasoning="Guess.")
    validated = validate_route_decision(
        decision, registry=registry, default_agent="general", allow_fallback=True
    )

    assert validated.agent == "general"
    assert validated.did_fallback is True
    assert validated.fallback_reason is not None
    assert "does_not_exist" in validated.fallback_reason
    assert validated.confidence == 0.8
    assert validated.reasoning == "Guess."


def test_validate_route_decision_allow_fallback_no_fallback_when_valid() -> None:
    registry = AgentRegistry()
    registry.register(AgentRegistration(name="general", description="General help."))
    registry.register(AgentRegistration(name="special", description="Specialized help."))

    decision = RouteDecision(agent="special", confidence=0.9, reasoning="Clear match.")
    validated = validate_route_decision(
        decision, registry=registry, default_agent="general", allow_fallback=True
    )

    assert validated.agent == "special"
    assert validated.did_fallback is False
    assert validated.fallback_reason is None


def test_validate_route_decision_allow_fallback_still_raises_on_invalid_default() -> None:
    registry = AgentRegistry()
    registry.register(AgentRegistration(name="special", description="Specialized help."))

    decision = RouteDecision(agent="does_not_exist", confidence=0.2, reasoning="Guess.")
    with pytest.raises(InvalidFallback):
        validate_route_decision(
            decision, registry=registry, default_agent="general", allow_fallback=True
        )


def test_validate_route_decision_confidence_threshold_triggers_fallback() -> None:
    registry = AgentRegistry()
    registry.register(AgentRegistration(name="general", description="General help."))
    registry.register(AgentRegistration(name="billing", description="Billing help."))

    decision = RouteDecision(agent="billing", confidence=0.15, reasoning="Not sure.")
    validated = validate_route_decision(
        decision, registry=registry, default_agent="general", confidence_threshold=0.5
    )

    assert validated.agent == "general"
    assert validated.did_fallback is True
    assert validated.fallback_reason is not None
    assert "0.15" in validated.fallback_reason
    assert "0.5" in validated.fallback_reason
    assert validated.confidence == 0.15


def test_validate_route_decision_confidence_threshold_no_fallback_when_above() -> None:
    registry = AgentRegistry()
    registry.register(AgentRegistration(name="general", description="General help."))
    registry.register(AgentRegistration(name="billing", description="Billing help."))

    decision = RouteDecision(agent="billing", confidence=0.9, reasoning="Clear match.")
    validated = validate_route_decision(
        decision, registry=registry, default_agent="general", confidence_threshold=0.5
    )

    assert validated.agent == "billing"
    assert validated.did_fallback is False


def test_validate_route_decision_confidence_threshold_skips_when_already_default() -> None:
    registry = AgentRegistry()
    registry.register(AgentRegistration(name="general", description="General help."))
    registry.register(AgentRegistration(name="billing", description="Billing help."))

    decision = RouteDecision(agent="general", confidence=0.1, reasoning="Fallback guess.")
    validated = validate_route_decision(
        decision, registry=registry, default_agent="general", confidence_threshold=0.5
    )

    assert validated.agent == "general"
    assert validated.did_fallback is False


def test_validate_route_decision_confidence_threshold_none_disables_check() -> None:
    registry = AgentRegistry()
    registry.register(AgentRegistration(name="general", description="General help."))
    registry.register(AgentRegistration(name="billing", description="Billing help."))

    decision = RouteDecision(agent="billing", confidence=0.1, reasoning="Low but allowed.")
    validated = validate_route_decision(
        decision, registry=registry, default_agent="general", confidence_threshold=None
    )

    assert validated.agent == "billing"
    assert validated.did_fallback is False


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
