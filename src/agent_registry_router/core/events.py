"""Routing observability events."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


class EventKind:
    """Constants for routing event kinds.

    Use these instead of raw strings when emitting or matching events.
    All values are plain strings, so ``EventKind.CLASSIFIER_RUN_START == "classifier_run_start"``
    holds true for backward compatibility.
    """

    PINNED_BYPASS: str = "pinned_bypass"
    PINNED_INVALID: str = "pinned_invalid"
    CLASSIFIER_RUN_START: str = "classifier_run_start"
    CLASSIFIER_RUN_SUCCESS: str = "classifier_run_success"
    DECISION_VALIDATED: str = "decision_validated"
    AGENT_RESOLVE_SUCCESS: str = "agent_resolve_success"
    AGENT_RESOLVE_FAILED: str = "agent_resolve_failed"
    AGENT_RUN_SUCCESS: str = "agent_run_success"
    AGENT_STREAM_CHUNK: str = "agent_stream_chunk"
    AGENT_STREAM_END: str = "agent_stream_end"
    AGENT_STREAM_EVENT: str = "agent_stream_event"
    AGENT_STREAM_RESPONSE: str = "agent_stream_response"


@dataclass(frozen=True)
class RoutingEvent:
    """Structured routing event emitted during classification/dispatch."""

    kind: str
    payload: dict[str, Any]
    error: BaseException | None = None


def emit_routing_event(
    kind: str,
    payload: dict[str, Any],
    *,
    error: BaseException | None = None,
    on_event: Callable[[RoutingEvent], None] | None = None,
    logger: logging.Logger,
) -> None:
    """Create and dispatch a RoutingEvent, swallowing hook failures."""
    event = RoutingEvent(kind=kind, payload=payload, error=error)
    if on_event:
        try:
            on_event(event)
        except Exception:
            logger.debug("Routing event hook failed", exc_info=True)
    if error:
        logger.debug("routing.%s error=%s payload=%s", kind, error, payload)
    else:
        logger.debug("routing.%s payload=%s", kind, payload)
