"""Routing observability events."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class RoutingEvent:
    """Structured routing event emitted during classification/dispatch."""

    kind: str
    payload: Dict[str, Any]
    error: Optional[BaseException] = None


