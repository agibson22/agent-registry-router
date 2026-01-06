from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Protocol

from pydantic import BaseModel

from agent_registry_router.core import (
    AgentRegistry,
    RouteDecision,
    ValidatedRouteDecision,
    validate_route_decision,
)


class AgentLike(Protocol):
    """PydanticAI-like agent interface (duck-typed).

    We intentionally avoid importing `pydantic_ai` at runtime so the core package
    stays lightweight. Any object with `await agent.run(message, deps=...)` works.
    """

    async def run(self, message: str, *, deps: Any) -> Any:  # pragma: no cover
        ...


def _normalize_name(name: str) -> str:
    return name.strip().lower()


def _extract_output(run_result: Any) -> Any:
    return getattr(run_result, "output", run_result)


def _coerce_route_decision(obj: Any) -> RouteDecision:
    if isinstance(obj, RouteDecision):
        return obj
    if isinstance(obj, dict):
        return RouteDecision(**obj)
    if isinstance(obj, BaseModel):
        data = obj.model_dump()
        if "agent" in data and "confidence" in data:
            # reasoning is optional in our core model
            return RouteDecision(
                agent=data["agent"],
                confidence=data["confidence"],
                reasoning=data.get("reasoning"),
            )
    # Last resort: attribute access
    agent = getattr(obj, "agent", None)
    confidence = getattr(obj, "confidence", None)
    reasoning = getattr(obj, "reasoning", None)
    if agent is None or confidence is None:
        raise TypeError("Classifier output must provide at least 'agent' and 'confidence'.")
    return RouteDecision(agent=agent, confidence=confidence, reasoning=reasoning)


@dataclass(frozen=True)
class DispatchResult:
    agent_name: str
    output: Any
    validated_decision: Optional[ValidatedRouteDecision]
    classifier_decision: Optional[RouteDecision]
    was_pinned: bool


class PydanticAIDispatcher:
    """Classifier + dispatch orchestrator for PydanticAI-style agents."""

    def __init__(
        self,
        *,
        registry: AgentRegistry,
        classifier_agent: AgentLike,
        get_agent: Callable[[str], Optional[AgentLike]],
        default_agent: str = "general",
    ) -> None:
        self._registry = registry
        self._classifier_agent = classifier_agent
        self._get_agent = get_agent
        self._default_agent = _normalize_name(default_agent)

    async def route_and_run(
        self,
        message: str,
        *,
        classifier_deps: Any,
        deps_for_agent: Callable[[str], Any],
        pinned_agent: Optional[str] = None,
    ) -> DispatchResult:
        """Route a message to an agent and run it.

        - If `pinned_agent` is provided, the dispatcher will **skip the classifier** and
          directly run that agent (even if it is not routable).
        - Otherwise, the dispatcher runs the classifier, validates its decision against
          `registry.routable_names()`, and dispatches to the selected agent (with fallback).
        """
        if pinned_agent:
            pinned = _normalize_name(pinned_agent)
            agent = self._get_agent(pinned)
            if agent is not None:
                deps = deps_for_agent(pinned)
                run_result = await agent.run(message, deps=deps)
                return DispatchResult(
                    agent_name=pinned,
                    output=_extract_output(run_result),
                    validated_decision=None,
                    classifier_decision=None,
                    was_pinned=True,
                )
            # If pinned is invalid, fall through to classifier routing.

        classifier_run = await self._classifier_agent.run(message, deps=classifier_deps)
        classifier_out = _extract_output(classifier_run)
        decision = _coerce_route_decision(classifier_out)

        validated = validate_route_decision(
            decision,
            registry=self._registry,
            default_agent=self._default_agent,
        )

        agent = self._get_agent(validated.agent)
        if agent is None:
            raise LookupError(f"Agent '{validated.agent}' not found (after validation).")

        deps = deps_for_agent(validated.agent)
        agent_run = await agent.run(message, deps=deps)

        return DispatchResult(
            agent_name=validated.agent,
            output=_extract_output(agent_run),
            validated_decision=validated,
            classifier_decision=decision,
            was_pinned=False,
        )


