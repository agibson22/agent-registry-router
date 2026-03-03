# Changelog

All notable changes to this repository will be documented in this file.

## v0.4.0 — 2026-03-03

### Added
- **Graceful fallback**: `validate_route_decision()` now accepts `allow_fallback=True`
  to swap to the default agent instead of raising `InvalidRouteDecision` when the
  classifier picks a non-routable agent. Sets `did_fallback` and `fallback_reason`
  on the returned `ValidatedRouteDecision`.
- **Confidence threshold**: `validate_route_decision()` now accepts
  `confidence_threshold` to fall back to the default agent when the classifier's
  confidence is below the threshold. Works for both LLM and FAISS classification paths.

### Changed
- Deduplicate shared routing helpers (`_coerce_route_decision`, `_normalize_name`,
  `_normalize_and_validate_pinned`) into `core.routing` — adapters now import from
  core instead of each defining their own copy.

## v0.3.0 — 2026-03-02

### Added
- **Eval suite**: benchmark classifier prompt quality across LLMs (GPT-4o-mini,
  Claude Haiku, Gemini Flash) and FAISS. Includes fixtures, runner, and report
  generator. `make eval` to run.
- **FAISS classifier**: embedding-based agent routing via cosine similarity as an
  alternative to LLM classification. Opt-in via `pip install "agent-registry-router[faiss]"`.
- **OpenAI Agents SDK adapter**: `OpenAIAgentsDispatcher` with `route_and_run()` and
  `route_and_stream()`. Duck-typed runner injection.
- **Google ADK adapter**: `GoogleADKDispatcher` with simplified `RunnerLike` protocol
  wrapping ADK's session-based runner.
- **Structured logging**: `StructuredLogger` — built-in JSON line handler for routing
  events. Use as `on_event` callback.
- **CONTRIBUTING.md**: contributor guide with setup, dev loop, and adapter guide.

### Changed
- Guard pinned-agent inputs by rejecting empty/whitespace values before routing.
- Clarify streaming classifier contract (must emit a final decision/output).
- Add a conservative upper bound for `pydantic` (<3.0) for forward-compat safety.

## v0.2.5

- Fix PyPI wheel contents to include the full `agent_registry_router` package (previous v0.2.4 wheel upload was missing modules like `agent_registry_router.core`).

## v0.2.4

- Add response/model-response streaming to the PydanticAI adapter via `PydanticAIDispatcher.route_and_stream_responses(...)` (exposes a `ResponseStreamSession` with access to the underlying streamed run handle).

## v0.2.3

- Add streaming dispatch support to the PydanticAI adapter: `PydanticAIDispatcher.route_and_stream(...)`.
- Add typed streaming payload `AgentStreamChunk` and export it from `agent_registry_router.adapters.pydantic_ai`.
- Preserve pinned bypass + fail-fast routing/validation behavior in streaming mode; emit `agent_stream_chunk` and `agent_stream_end` events.
- Add optional internally-consumed classifier streaming support (streamed classifier output is never yielded; only the chosen agent is streamed).
- Expand test coverage for streaming paths and common misconfigurations (lint/typecheck/coverage gates remain enforced).

## v0.2.0

- Change routing to fail-fast when classifier selects an unknown/non-routable agent or registry/default is invalid.
- Add typed exceptions (`RegistryError`, `RoutingError`, `InvalidRouteDecision`, `InvalidFallback`, `AgentNotFound`) and export them via `agent_registry_router.core`.
- Document behavior and error contracts in the README; add API contracts section.
- Enforce prompt stability: registration-order listing, routable-only inclusion, and error when no routable agents.
- Enforce agent description length cap (512 chars); optional `max_prompt_chars` guard for classifier prompt generation.
- Add observability hooks to `PydanticAIDispatcher` (`on_event`, logger) and emit structured routing events.
- Add lint/type/coverage gates (ruff, black, mypy, pytest-cov) and CI updates.
- Add package metadata (URLs, classifiers) and `py.typed` for typed distribution.

## v0.1.0

- Initial public release: registry-driven prompt builder, routing validation, and PydanticAI dispatcher with pinned bypass.


