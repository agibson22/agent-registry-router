from agent_registry_router.core import AgentRegistration, AgentRegistry, build_classifier_system_prompt


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


