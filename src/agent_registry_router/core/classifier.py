"""Embedding-based agent classifier using FAISS similarity search."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from agent_registry_router.core.exceptions import RegistryError
from agent_registry_router.core.registry import AgentRegistry
from agent_registry_router.core.routing import RouteDecision

EmbedFn = Callable[[list[str]], list[list[float]]]


class FaissClassifier:
    """Routes user messages to agents via embedding similarity instead of LLM calls.

    Embeds agent descriptions at initialization, then classifies incoming messages
    by finding the nearest agent description in the FAISS index.

    Args:
        registry: Agent registry with routable agents and descriptions.
        embed_fn: Callable that takes a list of strings and returns a list of
            embedding vectors. Bring your own — any provider works.
        default_agent: Fallback agent name for low-confidence matches.
        confidence_threshold: Minimum similarity score (0.0-1.0) to accept a
            match. Below this, routes to default_agent.
    """

    def __init__(
        self,
        *,
        registry: AgentRegistry,
        embed_fn: EmbedFn,
        default_agent: str = "general",
        confidence_threshold: float = 0.0,
    ) -> None:
        import faiss

        descriptions = registry.routable_descriptions()
        if not descriptions:
            raise RegistryError("No routable agents are registered.")

        self._agent_names: list[str] = list(descriptions.keys())
        self._default_agent = default_agent
        self._confidence_threshold = confidence_threshold
        self._embed_fn = embed_fn

        vectors = embed_fn(list(descriptions.values()))
        matrix = np.array(vectors, dtype=np.float32)
        faiss.normalize_L2(matrix)

        self._dimension = matrix.shape[1]
        self._index = faiss.IndexFlatIP(self._dimension)
        self._index.add(matrix)

    def classify(self, message: str) -> RouteDecision:
        """Classify a user message and return a RouteDecision."""
        import faiss

        query_vector = self._embed_fn([message])
        query_matrix = np.array(query_vector, dtype=np.float32)
        faiss.normalize_L2(query_matrix)

        scores, indices = self._index.search(query_matrix, 1)
        similarity = float(scores[0][0])
        best_index = int(indices[0][0])

        confidence = max(0.0, min(1.0, similarity))
        agent = self._agent_names[best_index]

        if confidence < self._confidence_threshold:
            agent = self._default_agent
            return RouteDecision(
                agent=agent,
                confidence=confidence,
                reasoning=(
                    f"Below confidence threshold "
                    f"({confidence:.2f} < {self._confidence_threshold}), "
                    f"routed to default."
                ),
            )

        return RouteDecision(
            agent=agent,
            confidence=confidence,
            reasoning=f"Nearest agent by embedding similarity ({confidence:.2f}).",
        )

    def classify_with_scores(self, message: str, top_k: int = 3) -> list[dict[str, Any]]:
        """Classify and return top-k agents with scores, for debugging."""
        import faiss

        k = min(top_k, len(self._agent_names))
        query_vector = self._embed_fn([message])
        query_matrix = np.array(query_vector, dtype=np.float32)
        faiss.normalize_L2(query_matrix)

        scores, indices = self._index.search(query_matrix, k)

        return [
            {
                "agent": self._agent_names[int(indices[0][i])],
                "similarity": float(scores[0][i]),
            }
            for i in range(k)
        ]
