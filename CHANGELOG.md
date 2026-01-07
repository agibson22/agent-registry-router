# Changelog

All notable changes to this repository will be documented in this file.

## Unreleased

- Change routing to fail-fast when classifier selects an unknown/non-routable agent or registry/default is invalid.
- Add typed exceptions (`RegistryError`, `RoutingError`, `InvalidRouteDecision`, `InvalidFallback`, `AgentNotFound`) and export them via `agent_registry_router.core`.
- Document behavior and error contracts in the README.
- Enforce prompt stability: registration-order listing, routable-only inclusion, and error when no routable agents.
- Enforce agent description length cap (512 chars).
- Add observability hooks to `PydanticAIDispatcher` (`on_event`, logger) and emit structured routing events.
- Add optional `max_prompt_chars` guard for classifier prompt generation.

## v0.1.0

- Initial public release: registry-driven prompt builder, routing validation, and PydanticAI dispatcher with pinned bypass.


