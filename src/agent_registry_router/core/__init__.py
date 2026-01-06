"""Core (framework-agnostic) primitives for agent-registry-router."""

from agent_registry_router.core.registry import AgentRegistration, AgentRegistry
from agent_registry_router.core.prompting import build_classifier_system_prompt
from agent_registry_router.core.routing import RouteDecision, ValidatedRouteDecision, validate_route_decision

__all__ = [
    "AgentRegistration",
    "AgentRegistry",
    "build_classifier_system_prompt",
    "RouteDecision",
    "ValidatedRouteDecision",
    "validate_route_decision",
]


