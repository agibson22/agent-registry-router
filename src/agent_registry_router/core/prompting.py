from __future__ import annotations

from typing import Optional

from agent_registry_router.core.exceptions import RegistryError
from agent_registry_router.core.registry import AgentRegistry


def build_classifier_system_prompt(
    registry: AgentRegistry,
    *,
    default_agent: str = "general",
    preamble: Optional[str] = None,
    extra_instructions: Optional[str] = None,
    max_prompt_chars: Optional[int] = None,
) -> str:
    """Build a classifier system prompt dynamically from registry agent descriptions."""
    descriptions = registry.routable_descriptions()
    if not descriptions:
        raise RegistryError("No routable agents are registered.")
    agent_sections = [f"**{name}**: {desc}" for name, desc in descriptions.items()]

    prompt_parts: list[str] = []
    base = (
        preamble.strip()
        if preamble
        else (
            "You are a query classifier that routes user messages to the appropriate agent. "
            "Analyze the user's intent and route to the best agent."
        )
    )
    if extra_instructions and extra_instructions.strip():
        base = base + " " + extra_instructions.strip()
    prompt_parts.append(base)

    prompt_parts.append("\n\n".join(agent_sections))

    prompt_parts.append(
        "Provide high confidence (0.8-1.0) for clear matches, "
        "medium confidence (0.5-0.7) for reasonable matches, "
        "and lower confidence (0.0-0.4) for uncertain cases. "
        f"When in doubt, route to '{default_agent}'."
    )

    prompt = "\n\n".join(prompt_parts).strip()
    if max_prompt_chars is not None and len(prompt) > max_prompt_chars:
        raise RegistryError(
            f"Classifier prompt exceeds max_prompt_chars={max_prompt_chars} (got {len(prompt)})."
        )
    return prompt


