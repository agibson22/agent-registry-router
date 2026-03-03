"""agent-registry-router: registry-driven LLM routing with optional framework adapters."""

from importlib.metadata import PackageNotFoundError, version

from agent_registry_router.core.events import EventKind, RoutingEvent
from agent_registry_router.core.exceptions import (
    AgentNotFound,
    AgentRegistryRouterError,
    InvalidFallback,
    InvalidRouteDecision,
    RegistryError,
    RoutingError,
)
from agent_registry_router.core.observability import StructuredLogger
from agent_registry_router.core.prompting import build_classifier_system_prompt
from agent_registry_router.core.registry import AgentRegistration, AgentRegistry
from agent_registry_router.core.routing import (
    RouteDecision,
    ValidatedRouteDecision,
    validate_route_decision,
)

__all__ = [
    "EventKind",
    "RoutingEvent",
    "AgentRegistryRouterError",
    "RegistryError",
    "RoutingError",
    "InvalidRouteDecision",
    "InvalidFallback",
    "AgentNotFound",
    "AgentRegistration",
    "AgentRegistry",
    "StructuredLogger",
    "build_classifier_system_prompt",
    "RouteDecision",
    "ValidatedRouteDecision",
    "validate_route_decision",
]

try:
    __version__ = version("agent-registry-router")
except PackageNotFoundError:
    __version__ = "0.0.0"

try:
    from agent_registry_router.core.classifier import FaissClassifier

    __all__ += ["FaissClassifier"]
except ImportError:
    pass
