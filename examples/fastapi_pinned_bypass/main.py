from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agent_registry_router.adapters.pydantic_ai import PydanticAIDispatcher
from agent_registry_router.core import AgentRegistration, AgentRegistry, RouteDecision


@dataclass
class FakeRunResult:
    output: Any


class ToyAgent:
    """A tiny agent that behaves like a PydanticAI agent (duck-typed)."""

    def __init__(self, name: str):
        self._name = name

    async def run(self, message: str, *, deps: Any) -> FakeRunResult:
        return FakeRunResult(
            {
                "agent": self._name,
                "message": message,
                "deps": deps,
                "answer": f"[{self._name}] handled: {message}",
            }
        )


class ToyClassifier:
    """A toy classifier that returns a RouteDecision, no LLM/API keys required."""

    async def run(self, message: str, *, deps: Any) -> FakeRunResult:
        msg = (message or "").lower()
        if "special" in msg:
            decision = RouteDecision(
                agent="special",
                confidence=0.9,
                reasoning="Keyword match: special.",
            )
        else:
            decision = RouteDecision(
                agent="general",
                confidence=0.6,
                reasoning="Default route.",
            )
        return FakeRunResult(decision)


class Session(BaseModel):
    id: UUID
    pinned_agent: str | None = None


class CreateSessionResponse(BaseModel):
    id: UUID


class PinRequest(BaseModel):
    pinned_agent: str | None = Field(default=None, description="If set, bypasses classifier.")


class MessageRequest(BaseModel):
    content: str


class MessageResponse(BaseModel):
    session_id: UUID
    used_agent: str
    was_pinned: bool
    output: Any
    classifier_decision: dict | None = None
    validated_decision: dict | None = None


app = FastAPI(title="agent-registry-router: pinned bypass example")


# In-memory sessions (demo only)
SESSIONS: dict[UUID, Session] = {}


# Registry and toy agents (demo only)
registry = AgentRegistry()
registry.register(
    AgentRegistration(
        name="general",
        description="Handles general queries.",
        routable=True,
    )
)
registry.register(
    AgentRegistration(
        name="special",
        description="Handles special queries.",
        routable=True,
    )
)
registry.register(
    AgentRegistration(
        name="internal_tool",
        description="Internal tool (not routable; can be pinned).",
        routable=False,
    )
)

AGENTS: dict[str, Any] = {
    "general": ToyAgent("general"),
    "special": ToyAgent("special"),
    "internal_tool": ToyAgent("internal_tool"),
}

dispatcher = PydanticAIDispatcher(
    registry=registry,
    classifier_agent=ToyClassifier(),
    get_agent=lambda name: AGENTS.get(name),
    default_agent="general",
)


@app.post("/sessions", response_model=CreateSessionResponse)
def create_session() -> CreateSessionResponse:
    sid = uuid4()
    SESSIONS[sid] = Session(id=sid, pinned_agent=None)
    return CreateSessionResponse(id=sid)


@app.post("/sessions/{session_id}/pin", response_model=Session)
def pin_session(session_id: UUID, req: PinRequest) -> Session:
    session = SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    session.pinned_agent = req.pinned_agent
    SESSIONS[session_id] = session
    return session


@app.post("/sessions/{session_id}/messages", response_model=MessageResponse)
async def send_message(session_id: UUID, req: MessageRequest) -> MessageResponse:
    session = SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    result = await dispatcher.route_and_run(
        req.content,
        classifier_deps={"session_id": str(session_id)},
        deps_for_agent=lambda agent_name: {
            "session_id": str(session_id),
            "agent": agent_name,
        },
        pinned_agent=session.pinned_agent,
    )

    return MessageResponse(
        session_id=session_id,
        used_agent=result.agent_name,
        was_pinned=result.was_pinned,
        output=result.output,
        classifier_decision=(
            result.classifier_decision.model_dump() if result.classifier_decision else None
        ),
        validated_decision=(
            result.validated_decision.model_dump() if result.validated_decision else None
        ),
    )
