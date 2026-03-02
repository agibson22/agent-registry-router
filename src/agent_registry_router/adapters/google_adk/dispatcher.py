"""Google ADK adapter for agent-registry-router.

Duck-typed dispatcher that works with the Google Agent Development Kit
without importing it at runtime. Any agent/runner matching the protocols
below works.

Google ADK uses an event-based runner (InMemoryRunner.run_async) with
sessions and Content objects. This adapter defines a simplified RunnerLike
protocol so users can wrap that ceremony in a callable that fits our
interface.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import BaseModel
from pydantic_core import ValidationError

from agent_registry_router.core import (
    AgentNotFound,
    AgentRegistry,
    InvalidRouteDecision,
    RouteDecision,
    RoutingEvent,
    ValidatedRouteDecision,
    validate_route_decision,
)


class GoogleADKAgentLike(Protocol):
    """Duck-typed protocol matching the Google ADK Agent interface."""

    @property
    def name(self) -> str: ...  # pragma: no cover


class RunResultLike(Protocol):
    """Duck-typed protocol for a run result."""

    @property
    def output(self) -> Any: ...  # pragma: no cover


class StreamEventLike(Protocol):
    """Duck-typed protocol for a stream event."""

    @property
    def author(self) -> str: ...  # pragma: no cover

    @property
    def content(self) -> Any: ...  # pragma: no cover


class RunnerLike(Protocol):
    """Simplified runner protocol for Google ADK.

    Users wrap ADK's InMemoryRunner (which requires sessions, user IDs,
    and Content objects) into this interface. The adapter doesn't know or
    care about ADK internals.
    """

    async def run(
        self, agent: Any, message: str, **kwargs: Any
    ) -> RunResultLike: ...  # pragma: no cover

    def run_stream(
        self, agent: Any, message: str, **kwargs: Any
    ) -> AsyncIterator[StreamEventLike]: ...  # pragma: no cover


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


@dataclass(frozen=True)
class DispatchResult:
    agent_name: str
    output: Any
    validated_decision: ValidatedRouteDecision | None
    classifier_decision: RouteDecision | None
    was_pinned: bool


@dataclass(frozen=True)
class StreamEvent:
    """A single streamed event from the selected agent."""

    agent_name: str
    event: Any
    validated_decision: ValidatedRouteDecision | None = None
    classifier_decision: RouteDecision | None = None
    was_pinned: bool = False
    event_index: int | None = None


class GoogleADKDispatcher:
    """Classifier + dispatch orchestrator for Google ADK agents.

    Uses a RunnerLike protocol that wraps ADK's InMemoryRunner. The user
    provides a runner that handles sessions, Content wrapping, and event
    extraction — this dispatcher only cares about classify → validate → dispatch.
    """

    def __init__(
        self,
        *,
        registry: AgentRegistry,
        classifier_agent: GoogleADKAgentLike,
        runner: RunnerLike,
        get_agent: Callable[[str], GoogleADKAgentLike | None],
        default_agent: str = "general",
        on_event: Callable[[RoutingEvent], None] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._registry = registry
        self._classifier_agent = classifier_agent
        self._runner = runner
        self._get_agent = get_agent
        self._default_agent = _normalize_name(default_agent)
        self._on_event = on_event
        self._logger = logger or logging.getLogger(__name__)

    def _emit(self, kind: str, payload: dict[str, Any], error: BaseException | None = None) -> None:
        event = RoutingEvent(kind=kind, payload=payload, error=error)
        if self._on_event:
            try:
                self._on_event(event)
            except Exception:
                self._logger.debug("Routing event hook failed", exc_info=True)
        if error:
            self._logger.debug("routing.%s error=%s payload=%s", kind, error, payload)
        else:
            self._logger.debug("routing.%s payload=%s", kind, payload)

    async def route_and_run(
        self,
        message: str,
        *,
        pinned_agent: str | None = None,
    ) -> DispatchResult:
        """Route a message to an agent and run it.

        If pinned_agent is provided, skips the classifier and runs that agent
        directly. Otherwise, classifies then dispatches.
        """
        if pinned_agent is not None:
            pinned = _normalize_and_validate_pinned(pinned_agent)
            agent = self._get_agent(pinned)
            if agent is not None:
                self._emit("pinned_bypass", {"agent": pinned, "message": message})
                run_result = await self._runner.run(agent, message)
                return DispatchResult(
                    agent_name=pinned,
                    output=run_result.output,
                    validated_decision=None,
                    classifier_decision=None,
                    was_pinned=True,
                )
            self._emit("pinned_invalid", {"pinned": pinned, "message": message})

        self._emit("classifier_run_start", {"message": message})
        classifier_result = await self._runner.run(self._classifier_agent, message)
        decision = _coerce_route_decision(classifier_result.output)
        self._emit(
            "classifier_run_success",
            {
                "message": message,
                "agent": decision.agent,
                "confidence": decision.confidence,
            },
        )

        validated = validate_route_decision(
            decision,
            registry=self._registry,
            default_agent=self._default_agent,
        )
        self._emit(
            "decision_validated",
            {
                "selected": validated.agent,
                "confidence": validated.confidence,
                "did_fallback": validated.did_fallback,
            },
        )

        agent = self._get_agent(validated.agent)
        if agent is None:
            error = AgentNotFound(f"Agent '{validated.agent}' not found (after validation).")
            self._emit(
                "agent_resolve_failed",
                {"agent": validated.agent},
                error=error,
            )
            raise error

        self._emit("agent_resolve_success", {"agent": validated.agent})
        agent_result = await self._runner.run(agent, message)

        self._emit("agent_run_success", {"agent": validated.agent})
        return DispatchResult(
            agent_name=validated.agent,
            output=agent_result.output,
            validated_decision=validated,
            classifier_decision=decision,
            was_pinned=False,
        )

    async def route_and_stream(
        self,
        message: str,
        *,
        pinned_agent: str | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Route a message to an agent and stream its events.

        The classifier is always run non-streamed; only the selected agent
        is streamed.
        """
        if pinned_agent is not None:
            pinned = _normalize_and_validate_pinned(pinned_agent)
            agent = self._get_agent(pinned)
            if agent is not None:
                self._emit("pinned_bypass", {"agent": pinned, "message": message})
                async for event in self._stream_agent(
                    agent,
                    pinned,
                    message,
                    validated_decision=None,
                    classifier_decision=None,
                    was_pinned=True,
                ):
                    yield event
                return
            self._emit("pinned_invalid", {"pinned": pinned, "message": message})

        self._emit("classifier_run_start", {"message": message})
        classifier_result = await self._runner.run(self._classifier_agent, message)
        decision = _coerce_route_decision(classifier_result.output)
        self._emit(
            "classifier_run_success",
            {
                "message": message,
                "agent": decision.agent,
                "confidence": decision.confidence,
            },
        )

        validated = validate_route_decision(
            decision,
            registry=self._registry,
            default_agent=self._default_agent,
        )
        self._emit(
            "decision_validated",
            {
                "selected": validated.agent,
                "confidence": validated.confidence,
                "did_fallback": validated.did_fallback,
            },
        )

        agent = self._get_agent(validated.agent)
        if agent is None:
            error = AgentNotFound(f"Agent '{validated.agent}' not found (after validation).")
            self._emit(
                "agent_resolve_failed",
                {"agent": validated.agent},
                error=error,
            )
            raise error

        self._emit("agent_resolve_success", {"agent": validated.agent})
        async for event in self._stream_agent(
            agent,
            validated.agent,
            message,
            validated_decision=validated,
            classifier_decision=decision,
            was_pinned=False,
        ):
            yield event

    async def _stream_agent(
        self,
        agent: GoogleADKAgentLike,
        agent_name: str,
        message: str,
        *,
        validated_decision: ValidatedRouteDecision | None,
        classifier_decision: RouteDecision | None,
        was_pinned: bool,
    ) -> AsyncIterator[StreamEvent]:
        """Stream events from a single agent via runner.run_stream()."""
        events_emitted = 0
        first = True

        try:
            async for raw_event in self._runner.run_stream(agent, message):
                payload: dict[str, Any] = {
                    "agent": agent_name,
                    "event_index": events_emitted,
                    "event_type": type(raw_event).__name__,
                }
                self._emit("agent_stream_event", payload)

                if first:
                    yield StreamEvent(
                        agent_name=agent_name,
                        event=raw_event,
                        validated_decision=validated_decision,
                        classifier_decision=classifier_decision,
                        was_pinned=was_pinned,
                        event_index=events_emitted,
                    )
                    first = False
                else:
                    yield StreamEvent(
                        agent_name=agent_name,
                        event=raw_event,
                        was_pinned=was_pinned,
                        event_index=events_emitted,
                    )
                events_emitted += 1
        finally:
            self._emit(
                "agent_stream_end",
                {"agent": agent_name, "events_emitted": events_emitted},
            )
            self._emit("agent_run_success", {"agent": agent_name})
