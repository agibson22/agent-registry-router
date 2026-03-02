"""Built-in structured logging handler for routing events."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from datetime import datetime, timezone
from typing import IO, Any

from agent_registry_router.core.events import RoutingEvent


class StructuredLogger:
    """Logs RoutingEvents as JSON lines. Use as an on_event callback.

    Args:
        sink: Where to write JSON lines. Accepts a file-like object (default:
            sys.stdout) or any callable that takes a string.
    """

    def __init__(
        self,
        sink: IO[str] | Callable[[str], Any] | None = None,
    ) -> None:
        self._file_sink: IO[str] | None = None
        self._fn_sink: Callable[[str], Any] | None = None

        if sink is None:
            self._file_sink = sys.stdout
        elif hasattr(sink, "write"):
            self._file_sink = sink  # type: ignore[assignment]
        else:
            self._fn_sink = sink  # type: ignore[assignment]

    def __call__(self, event: RoutingEvent) -> None:
        record: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event.kind,
            **event.payload,
        }
        if event.error is not None:
            record["error"] = str(event.error)
            record["error_type"] = type(event.error).__name__

        line = json.dumps(record, default=str)

        if self._fn_sink is not None:
            self._fn_sink(line)
        elif self._file_sink is not None:
            self._file_sink.write(line + "\n")
            self._file_sink.flush()
