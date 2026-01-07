"""Typed exceptions for agent-registry-router."""


class AgentRegistryRouterError(Exception):
    """Base exception for agent-registry-router."""


class RegistryError(AgentRegistryRouterError):
    """Registry-related errors (e.g., invalid names, missing registry state)."""


class RoutingError(AgentRegistryRouterError):
    """Routing/validation errors emitted during decision handling."""


class InvalidRouteDecision(RoutingError):
    """Raised when a classifier decision is structurally or semantically invalid."""


class InvalidFallback(RoutingError):
    """Raised when fallback cannot proceed (e.g., no routable/default agent)."""


class AgentNotFound(RoutingError):
    """Raised when a resolved agent cannot be located for dispatch."""

