from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from pydantic import BaseModel, Field


def _normalize_agent_name(name: str) -> str:
    if not name or not name.strip():
        raise ValueError("Agent name cannot be empty")
    return name.strip().lower()


class AgentRegistration(BaseModel):
    """Registration metadata for an agent.

    Note: the core package does not assume any agent runtime/framework.
    Framework adapters can store runtime-specific objects elsewhere.
    """

    name: str = Field(description="Stable agent identifier (normalized to lowercase).")
    description: str = Field(description="Short description used to build classifier prompts.")
    routable: bool = Field(
        default=True,
        description="If false, excluded from classifier prompt building and routing targets.",
    )
    deps_type: Any = Field(
        default=None,
        description="Optional dependency type/metadata for adapters (core does not instantiate deps).",
    )

    def model_post_init(self, __context: Any) -> None:
        self.name = _normalize_agent_name(self.name)


class AgentRegistry:
    """Registry of agent metadata used for prompt building and routing validation."""

    def __init__(self) -> None:
        self._agents: Dict[str, AgentRegistration] = {}

    @classmethod
    def from_descriptions(cls, descriptions: Mapping[str, str], *, routable: bool = True) -> "AgentRegistry":
        """Convenience constructor for building a registry from name->description mappings."""
        registry = cls()
        for name, description in descriptions.items():
            registry.register(AgentRegistration(name=name, description=description, routable=routable))
        return registry

    def register(self, registration: AgentRegistration) -> None:
        """Register (or overwrite) an agent registration."""
        self._agents[_normalize_agent_name(registration.name)] = registration

    def get(self, name: str) -> Optional[AgentRegistration]:
        """Get a registration by name."""
        return self._agents.get(_normalize_agent_name(name))

    def all_names(self) -> List[str]:
        return list(self._agents.keys())

    def routable_names(self) -> List[str]:
        return [name for name, reg in self._agents.items() if reg.routable]

    def descriptions(self) -> Dict[str, str]:
        return {name: reg.description for name, reg in self._agents.items()}

    def routable_descriptions(self) -> Dict[str, str]:
        return {name: reg.description for name, reg in self._agents.items() if reg.routable}


