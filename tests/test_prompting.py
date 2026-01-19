import pytest

from agent_registry_router.core import (
    AgentRegistration,
    AgentRegistry,
    RegistryError,
    build_classifier_system_prompt,
)


def test_build_classifier_system_prompt_includes_routable_agents_only() -> None:
    registry = AgentRegistry()
    registry.register(
        AgentRegistration(
            name="general",
            description="General help.",
            routable=True,
        )
    )
    registry.register(
        AgentRegistration(
            name="internal_tool",
            description="Internal tool (should not appear).",
            routable=False,
        )
    )

    prompt = build_classifier_system_prompt(registry)
    assert "**general**: General help." in prompt
    assert "internal_tool" not in prompt


def test_build_classifier_system_prompt_uses_custom_preamble() -> None:
    registry = AgentRegistry()
    registry.register(AgentRegistration(name="general", description="General help."))

    preamble = "Important: The UI uses @mentions."
    prompt = build_classifier_system_prompt(registry, preamble=preamble)

    assert prompt.startswith(preamble)
    assert "**general**: General help." in prompt


def test_build_classifier_system_prompt_appends_extra_instructions() -> None:
    registry = AgentRegistry()
    registry.register(AgentRegistration(name="general", description="General help."))

    preamble = "You are a query classifier."
    extra = "Important: The UI uses @mentions."
    prompt = build_classifier_system_prompt(registry, preamble=preamble, extra_instructions=extra)

    assert prompt.startswith(preamble + " " + extra)


def test_build_classifier_system_prompt_preserves_registration_order() -> None:
    registry = AgentRegistry()
    registry.register(AgentRegistration(name="alpha", description="A"))
    registry.register(AgentRegistration(name="beta", description="B"))
    registry.register(AgentRegistration(name="gamma", description="C"))

    prompt = build_classifier_system_prompt(registry)
    assert prompt.index("**alpha**") < prompt.index("**beta**") < prompt.index("**gamma**")


def test_build_classifier_system_prompt_raises_when_no_routable_agents() -> None:
    registry = AgentRegistry()
    with pytest.raises(RegistryError):
        build_classifier_system_prompt(registry)


def test_agent_description_length_limit() -> None:
    registry = AgentRegistry()
    too_long = "x" * 513
    with pytest.raises(RegistryError):
        registry.register(AgentRegistration(name="longy", description=too_long))


def test_build_classifier_system_prompt_respects_max_chars() -> None:
    registry = AgentRegistry()
    registry.register(AgentRegistration(name="general", description="hi"))
    prompt = build_classifier_system_prompt(registry, max_prompt_chars=1000)
    assert len(prompt) <= 1000

    with pytest.raises(RegistryError):
        build_classifier_system_prompt(registry, max_prompt_chars=10)
