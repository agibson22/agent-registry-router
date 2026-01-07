"""Core (framework-agnostic) primitives for agent-registry-router."""

from agent_registry_router.core.exceptions import (
    AgentNotFound,
    AgentRegistryRouterError,
    InvalidFallback,
    InvalidRouteDecision,
    RegistryError,
    RoutingError,
)
from agent_registry_router.core.registry import AgentRegistration, AgentRegistry
from agent_registry_router.core.prompting import build_classifier_system_prompt
from agent_registry_router.core.routing import RouteDecision, ValidatedRouteDecision, validate_route_decision

__all__ = [
    "AgentRegistryRouterError",
    "RegistryError",
    "RoutingError",
    "InvalidRouteDecision",
    "InvalidFallback",
    "AgentNotFound",
    "AgentRegistration",
    "AgentRegistry",
    "build_classifier_system_prompt",
    "RouteDecision",
    "ValidatedRouteDecision",
    "validate_route_decision",
]


