from __future__ import annotations

from typing import Optional

from agent_registry_router.core.registry import AgentRegistry


def build_classifier_system_prompt(
    registry: AgentRegistry,
    *,
    default_agent: str = "general",
    preamble: Optional[str] = None,
) -> str:
    """Build a classifier system prompt dynamically from registry agent descriptions."""
    descriptions = registry.routable_descriptions()
    agent_sections = [f"**{name}**: {desc}" for name, desc in descriptions.items()]

    prompt_parts: list[str] = []
    if preamble:
        prompt_parts.append(preamble.strip())
    else:
        prompt_parts.append(
            "You are a query classifier that routes user messages to the appropriate agent. "
            "Analyze the user's intent and route to the best agent."
        )

    if agent_sections:
        prompt_parts.append("\n\n".join(agent_sections))
    else:
        prompt_parts.append("No routable agents are registered.")

    prompt_parts.append(
        "Provide high confidence (0.8-1.0) for clear matches, "
        "medium confidence (0.5-0.7) for reasonable matches, "
        "and lower confidence (0.0-0.4) for uncertain cases. "
        f"When in doubt, route to '{default_agent}'."
    )

    return "\n\n".join(prompt_parts).strip()


