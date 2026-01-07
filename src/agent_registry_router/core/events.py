"""Routing observability events."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RoutingEvent:
    """Structured routing event emitted during classification/dispatch."""

    kind: str
    payload: dict[str, Any]
    error: BaseException | None = None
