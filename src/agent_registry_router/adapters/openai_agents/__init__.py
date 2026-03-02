"""OpenAI Agents SDK adapter (optional dependency)."""

from agent_registry_router.adapters.openai_agents.dispatcher import (
    DispatchResult,
    OpenAIAgentsDispatcher,
    StreamEvent,
)

__all__ = [
    "DispatchResult",
    "OpenAIAgentsDispatcher",
    "StreamEvent",
]
