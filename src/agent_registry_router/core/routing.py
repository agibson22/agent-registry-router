from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field
from pydantic_core import ValidationError

from agent_registry_router.core.exceptions import InvalidFallback, InvalidRouteDecision
from agent_registry_router.core.registry import AgentRegistry, _normalize_agent_name


class RouteDecision(BaseModel):
    """Structured routing decision emitted by a classifier."""

    agent: str = Field(description="Chosen agent name (must be a routable registered agent).")
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in the routing decision, 0.0-1.0.",
    )
    reasoning: str | None = Field(default=None, description="Short explanation for the choice.")

    def model_post_init(self, __context: Any) -> None:  # type: ignore[override]
        self.agent = _normalize_agent_name(self.agent)


class ValidatedRouteDecision(RouteDecision):
    """RouteDecision after validation against a registry."""

    did_fallback: bool = Field(
        default=False,
        description="True if the decision was changed to the default agent due to validation.",
    )
    fallback_reason: str | None = Field(
        default=None,
        description="Reason the decision fell back to the default agent.",
    )


def validate_route_decision(
    decision: RouteDecision,
    *,
    registry: AgentRegistry,
    default_agent: str = "general",
) -> ValidatedRouteDecision:
    """Validate the route decision against routable agents; fail-fast on invalid cases."""
    routable = set(registry.routable_names())
    normalized_default = _normalize_agent_name(default_agent)

    if not routable:
        raise InvalidFallback("No routable agents registered.")

    if normalized_default not in routable:
        raise InvalidFallback(
            f"Default agent '{default_agent}' is not a routable registered agent."
        )

    if decision.agent not in routable:
        raise InvalidRouteDecision(f"Agent '{decision.agent}' is not a routable registered agent.")

    return ValidatedRouteDecision(**decision.model_dump(), did_fallback=False)


def _normalize_name(name: str) -> str:
    return name.strip().lower()


def _normalize_and_validate_pinned(name: str) -> str:
    trimmed = name.strip()
    if not trimmed:
        raise InvalidRouteDecision("Pinned agent cannot be empty or whitespace.")
    return trimmed.lower()


def _coerce_route_decision(obj: Any) -> RouteDecision:
    if isinstance(obj, RouteDecision):
        return obj
    if isinstance(obj, dict):
        try:
            return RouteDecision(**obj)
        except ValidationError as exc:
            raise InvalidRouteDecision(str(exc)) from exc
    if isinstance(obj, BaseModel):
        data = obj.model_dump()
        if "agent" in data and "confidence" in data:
            return RouteDecision(
                agent=data["agent"],
                confidence=data["confidence"],
                reasoning=data.get("reasoning"),
            )
    agent = getattr(obj, "agent", None)
    confidence = getattr(obj, "confidence", None)
    reasoning = getattr(obj, "reasoning", None)
    if agent is None or confidence is None:
        raise InvalidRouteDecision(
            "Classifier output must provide at least 'agent' and 'confidence'."
        )
    return RouteDecision(agent=agent, confidence=confidence, reasoning=reasoning)
