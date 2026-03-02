"""Google ADK adapter (optional dependency)."""

from agent_registry_router.adapters.google_adk.dispatcher import (
    DispatchResult,
    GoogleADKDispatcher,
    StreamEvent,
)

__all__ = [
    "DispatchResult",
    "GoogleADKDispatcher",
    "StreamEvent",
]
