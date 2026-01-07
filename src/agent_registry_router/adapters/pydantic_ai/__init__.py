"""PydanticAI adapter (optional dependency)."""

from agent_registry_router.adapters.pydantic_ai.dispatcher import (
    AgentResponseStreamItem,
    AgentStreamChunk,
    DispatchResult,
    PydanticAIDispatcher,
    ResponseStreamSession,
)

__all__ = [
    "AgentResponseStreamItem",
    "AgentStreamChunk",
    "DispatchResult",
    "PydanticAIDispatcher",
    "ResponseStreamSession",
]
