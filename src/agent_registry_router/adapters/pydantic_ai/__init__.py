"""PydanticAI adapter (optional dependency)."""

from agent_registry_router.adapters.pydantic_ai.dispatcher import (
    AgentStreamChunk,
    DispatchResult,
    PydanticAIDispatcher,
)

__all__ = ["AgentStreamChunk", "DispatchResult", "PydanticAIDispatcher"]
