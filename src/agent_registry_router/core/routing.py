from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from agent_registry_router.core.registry import AgentRegistry, _normalize_agent_name


class RouteDecision(BaseModel):
    """Structured routing decision emitted by a classifier."""

    agent: str = Field(description="Chosen agent name (must be a routable registered agent).")
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in the routing decision, 0.0-1.0.",
    )
    reasoning: Optional[str] = Field(default=None, description="Short explanation for the choice.")

    def model_post_init(self, __context) -> None:  # type: ignore[override]
        self.agent = _normalize_agent_name(self.agent)


class ValidatedRouteDecision(RouteDecision):
    """RouteDecision after validation against a registry."""

    did_fallback: bool = Field(
        default=False,
        description="True if the decision was changed to the default agent due to validation.",
    )
    fallback_reason: Optional[str] = Field(
        default=None,
        description="Reason the decision fell back to the default agent.",
    )


def validate_route_decision(
    decision: RouteDecision,
    *,
    registry: AgentRegistry,
    default_agent: str = "general",
) -> ValidatedRouteDecision:
    """Validate the route decision against routable agents, falling back if invalid."""
    routable = set(registry.routable_names())
    normalized_default = _normalize_agent_name(default_agent)

    if decision.agent in routable:
        return ValidatedRouteDecision(**decision.model_dump(), did_fallback=False)

    fallback_agent = normalized_default if normalized_default in routable else (next(iter(routable)) if routable else normalized_default)
    return ValidatedRouteDecision(
        agent=fallback_agent,
        confidence=max(0.0, min(1.0, decision.confidence - 0.3)),
        reasoning=decision.reasoning,
        did_fallback=True,
        fallback_reason=f"Agent '{decision.agent}' is not a routable registered agent.",
    )


