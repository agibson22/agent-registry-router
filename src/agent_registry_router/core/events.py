"""Routing observability events."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


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
