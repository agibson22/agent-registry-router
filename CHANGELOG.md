# Changelog

All notable changes to this repository will be documented in this file.

## Unreleased

- Change routing to fail-fast when classifier selects an unknown/non-routable agent or registry/default is invalid.
- Add typed exceptions (`RegistryError`, `RoutingError`, `InvalidRouteDecision`, `InvalidFallback`, `AgentNotFound`) and export them via `agent_registry_router.core`.
- Document behavior and error contracts in the README.

## v0.1.0

- Initial public release: registry-driven prompt builder, routing validation, and PydanticAI dispatcher with pinned bypass.


