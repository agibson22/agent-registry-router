from __future__ import annotations

import logging
from collections.abc import AsyncIterable, AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager
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


class AgentLike(Protocol):
    """PydanticAI-like agent interface (duck-typed).

    We intentionally avoid importing `pydantic_ai` at runtime so the core package
    stays lightweight. Any object with `await agent.run(message, deps=...)` works.
    """

    async def run(self, message: str, *, deps: Any) -> Any:  # pragma: no cover
        ...


class StreamableAgentLike(Protocol):
    """Optional PydanticAI-like streaming interface (duck-typed)."""

    def run_stream(  # pragma: no cover
        self, message: str, *, deps: Any
    ) -> AbstractAsyncContextManager[Any]: ...


class StreamEventsAgentLike(Protocol):
    """Optional PydanticAI-like event streaming interface (duck-typed)."""

    def run_stream_events(  # pragma: no cover
        self, message: str, *, deps: Any
    ) -> AsyncIterable[Any]: ...


def _normalize_name(name: str) -> str:
    return name.strip().lower()


def _extract_output(run_result: Any) -> Any:
    return getattr(run_result, "output", run_result)


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
class AgentStreamChunk:
    agent_name: str
    chunk: Any
    validated_decision: ValidatedRouteDecision | None = None
    classifier_decision: RouteDecision | None = None
    was_pinned: bool = False
    chunk_index: int | None = None


class PydanticAIDispatcher:
    """Classifier + dispatch orchestrator for PydanticAI-style agents."""

    def __init__(
        self,
        *,
        registry: AgentRegistry,
        classifier_agent: AgentLike,
        get_agent: Callable[[str], AgentLike | None],
        default_agent: str = "general",
        on_event: Callable[[RoutingEvent], None] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._registry = registry
        self._classifier_agent = classifier_agent
        self._get_agent = get_agent
        self._default_agent = _normalize_name(default_agent)
        self._on_event = on_event
        self._logger = logger or logging.getLogger(__name__)

    def _emit(
        self, kind: str, payload: dict, error: BaseException | None = None
    ) -> None:
        event = RoutingEvent(kind=kind, payload=payload, error=error)
        if self._on_event:
            try:
                self._on_event(event)
            except Exception:
                # Observability hooks should not break routing.
                self._logger.debug("Routing event hook failed", exc_info=True)
        if error:
            self._logger.debug("routing.%s error=%s payload=%s", kind, error, payload)
        else:
            self._logger.debug("routing.%s payload=%s", kind, payload)

    async def route_and_run(
        self,
        message: str,
        *,
        classifier_deps: Any,
        deps_for_agent: Callable[[str], Any],
        pinned_agent: str | None = None,
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
                self._emit("pinned_bypass", {"agent": pinned, "message": message})
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
            self._emit("pinned_invalid", {"pinned": pinned, "message": message})

        self._emit("classifier_run_start", {"message": message})
        classifier_run = await self._classifier_agent.run(message, deps=classifier_deps)
        classifier_out = _extract_output(classifier_run)
        decision = _coerce_route_decision(classifier_out)
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
            error = AgentNotFound(
                f"Agent '{validated.agent}' not found (after validation)."
            )
            self._emit("agent_resolve_failed", {"agent": validated.agent}, error=error)
            raise error

        self._emit("agent_resolve_success", {"agent": validated.agent})
        deps = deps_for_agent(validated.agent)
        agent_run = await agent.run(message, deps=deps)

        self._emit("agent_run_success", {"agent": validated.agent})
        return DispatchResult(
            agent_name=validated.agent,
            output=_extract_output(agent_run),
            validated_decision=validated,
            classifier_decision=decision,
            was_pinned=False,
        )

    async def route_and_stream(
        self,
        message: str,
        *,
        classifier_deps: Any,
        deps_for_agent: Callable[[str], Any],
        pinned_agent: str | None = None,
        stream_classifier: bool = False,
    ) -> AsyncIterator[AgentStreamChunk]:
        """Route a message to an agent and stream its output.

        The routing semantics mirror `route_and_run`, but only the **selected agent**
        is streamed outward. If `stream_classifier` is enabled, classifier streaming is
        consumed internally to completion (never yielded).
        """
        if pinned_agent:
            pinned = _normalize_name(pinned_agent)
            agent = self._get_agent(pinned)
            if agent is not None:
                self._emit("pinned_bypass", {"agent": pinned, "message": message})
                deps = deps_for_agent(pinned)

                run_stream = getattr(agent, "run_stream", None)
                if run_stream is None:
                    raise TypeError(
                        f"Agent '{pinned}' does not support streaming (missing run_stream)."
                    )

                chunks_emitted = 0
                async with run_stream(message, deps=deps) as streamed:
                    stream_text = getattr(streamed, "stream_text", None)
                    if stream_text is None:
                        raise TypeError(
                            f"Agent '{pinned}' stream result does not support text streaming "
                            "(missing stream_text)."
                        )
                    try:
                        iterator = stream_text(delta=True)
                    except TypeError:
                        iterator = stream_text()

                    async for chunk in iterator:
                        payload: dict[str, Any] = {
                            "agent": pinned,
                            "chunk_index": chunks_emitted,
                            "chunk_type": type(chunk).__name__,
                        }
                        if isinstance(chunk, str):
                            payload["text_len"] = len(chunk)
                        self._emit("agent_stream_chunk", payload)
                        yield AgentStreamChunk(
                            agent_name=pinned,
                            chunk=chunk,
                            was_pinned=True,
                            chunk_index=chunks_emitted,
                        )
                        chunks_emitted += 1

                self._emit(
                    "agent_stream_end",
                    {"agent": pinned, "chunks_emitted": chunks_emitted},
                )
                self._emit("agent_run_success", {"agent": pinned})
                return
            # If pinned is invalid, fall through to classifier routing.
            self._emit("pinned_invalid", {"pinned": pinned, "message": message})

        decision = await self._run_classifier_decision(
            message,
            classifier_deps=classifier_deps,
            stream_classifier=stream_classifier,
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
            error = AgentNotFound(
                f"Agent '{validated.agent}' not found (after validation)."
            )
            self._emit("agent_resolve_failed", {"agent": validated.agent}, error=error)
            raise error

        self._emit("agent_resolve_success", {"agent": validated.agent})
        deps = deps_for_agent(validated.agent)

        run_stream = getattr(agent, "run_stream", None)
        if run_stream is None:
            raise TypeError(
                f"Agent '{validated.agent}' does not support streaming (missing run_stream)."
            )

        chunks_emitted = 0
        first = True
        async with run_stream(message, deps=deps) as streamed:
            stream_text = getattr(streamed, "stream_text", None)
            if stream_text is None:
                raise TypeError(
                    f"Agent '{validated.agent}' stream result does not support text streaming "
                    "(missing stream_text)."
                )
            try:
                iterator = stream_text(delta=True)
            except TypeError:
                iterator = stream_text()

            async for chunk in iterator:
                payload = {
                    "agent": validated.agent,
                    "chunk_index": chunks_emitted,
                    "chunk_type": type(chunk).__name__,
                }
                if isinstance(chunk, str):
                    payload["text_len"] = len(chunk)
                self._emit("agent_stream_chunk", payload)

                if first:
                    yield AgentStreamChunk(
                        agent_name=validated.agent,
                        chunk=chunk,
                        validated_decision=validated,
                        classifier_decision=decision,
                        was_pinned=False,
                        chunk_index=chunks_emitted,
                    )
                    first = False
                else:
                    yield AgentStreamChunk(
                        agent_name=validated.agent,
                        chunk=chunk,
                        was_pinned=False,
                        chunk_index=chunks_emitted,
                    )
                chunks_emitted += 1

        self._emit(
            "agent_stream_end",
            {"agent": validated.agent, "chunks_emitted": chunks_emitted},
        )
        self._emit("agent_run_success", {"agent": validated.agent})

    async def _run_classifier_decision(
        self,
        message: str,
        *,
        classifier_deps: Any,
        stream_classifier: bool,
    ) -> RouteDecision:
        self._emit("classifier_run_start", {"message": message})

        if stream_classifier:
            decision = await self._run_classifier_decision_streaming(
                message, classifier_deps=classifier_deps
            )
        else:
            classifier_run = await self._classifier_agent.run(
                message, deps=classifier_deps
            )
            classifier_out = _extract_output(classifier_run)
            decision = _coerce_route_decision(classifier_out)

        self._emit(
            "classifier_run_success",
            {
                "message": message,
                "agent": decision.agent,
                "confidence": decision.confidence,
            },
        )
        return decision

    async def _run_classifier_decision_streaming(
        self,
        message: str,
        *,
        classifier_deps: Any,
    ) -> RouteDecision:
        """Consume a streaming classifier run internally to completion.

        This never yields classifier chunks; it returns the final RouteDecision output.
        """
        classifier = self._classifier_agent

        run_stream_events = getattr(classifier, "run_stream_events", None)
        if run_stream_events is not None:
            last_result: Any | None = None
            async for event in run_stream_events(message, deps=classifier_deps):
                # PydanticAI uses AgentRunResultEvent(result=AgentRunResult(output=...))
                maybe_result = getattr(event, "result", None)
                if maybe_result is not None:
                    last_result = maybe_result
            if last_result is None:
                raise InvalidRouteDecision(
                    "Streaming classifier produced no final result event."
                )
            classifier_out = _extract_output(last_result)
            return _coerce_route_decision(classifier_out)

        run_stream = getattr(classifier, "run_stream", None)
        if run_stream is None:
            # Fall back to non-streaming classifier if it doesn't support streaming.
            classifier_run = await classifier.run(message, deps=classifier_deps)
            classifier_out = _extract_output(classifier_run)
            return _coerce_route_decision(classifier_out)

        # PydanticAI `run_stream` is an async context manager returning a StreamedRunResult.
        async with run_stream(message, deps=classifier_deps) as streamed:
            stream_output = getattr(streamed, "stream_output", None)
            stream_text = getattr(streamed, "stream_text", None)

            final: Any | None = None
            if stream_output is not None:
                maybe_iter = stream_output()
                if hasattr(maybe_iter, "__aiter__"):
                    async for item in maybe_iter:
                        final = item
                else:
                    for item in maybe_iter:
                        final = item
            elif stream_text is not None:
                try:
                    maybe_iter = stream_text(delta=True)
                except TypeError:
                    maybe_iter = stream_text()
                async for item in maybe_iter:
                    final = item
            else:
                raise InvalidRouteDecision(
                    "Streaming classifier does not provide stream_output() or stream_text()."
                )

            if final is None:
                raise InvalidRouteDecision("Streaming classifier produced no output.")
            return _coerce_route_decision(final)
