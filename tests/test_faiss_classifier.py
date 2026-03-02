from collections.abc import Callable

import pytest

from agent_registry_router.core import (
    AgentRegistration,
    AgentRegistry,
    RegistryError,
)
from agent_registry_router.core.classifier import FaissClassifier


def _make_embed_fn(
    dimension: int = 8,
) -> Callable[[list[str]], list[list[float]]]:
    """Create a deterministic embedding function for testing.

    Maps text to a fixed vector based on keyword matching. Texts containing
    'billing' or 'payment' get a vector near [1,0,0,...], 'technical' or 'error'
    near [0,1,0,...], etc. This lets us verify FAISS picks the right agent
    without calling a real embedding API.
    """
    keywords_to_axis = {
        ("billing", "payment", "invoice", "charge", "refund"): 0,
        ("technical", "error", "bug", "crash", "outage"): 1,
        ("sales", "pricing", "upgrade", "plan", "buy"): 2,
        ("account", "password", "login", "profile", "access"): 3,
    }

    def embed_fn(texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vec = [0.1] * dimension
            lower = text.lower()
            for keywords, axis in keywords_to_axis.items():
                if any(kw in lower for kw in keywords):
                    vec[axis] = 1.0
                    break
            vectors.append(vec)
        return vectors

    return embed_fn


def _make_registry() -> AgentRegistry:
    registry = AgentRegistry()
    registry.register(
        AgentRegistration(
            name="billing", description="Handles billing and payment issues."
        )
    )
    registry.register(
        AgentRegistration(
            name="technical", description="Handles technical errors and bugs."
        )
    )
    registry.register(
        AgentRegistration(
            name="sales", description="Handles pricing and upgrade questions."
        )
    )
    registry.register(
        AgentRegistration(name="general", description="Handles general inquiries.")
    )
    return registry


def test_classify_routes_to_correct_agent() -> None:
    classifier = FaissClassifier(
        registry=_make_registry(),
        embed_fn=_make_embed_fn(),
    )

    decision = classifier.classify("I was charged twice on my credit card")
    assert decision.agent == "billing"
    assert 0.0 <= decision.confidence <= 1.0
    assert decision.reasoning is not None


def test_classify_technical_query() -> None:
    classifier = FaissClassifier(
        registry=_make_registry(),
        embed_fn=_make_embed_fn(),
    )

    decision = classifier.classify("The app crashes with an error message")
    assert decision.agent == "technical"


def test_classify_sales_query() -> None:
    classifier = FaissClassifier(
        registry=_make_registry(),
        embed_fn=_make_embed_fn(),
    )

    decision = classifier.classify("How much does the upgrade plan cost?")
    assert decision.agent == "sales"


def test_classify_returns_valid_route_decision() -> None:
    classifier = FaissClassifier(
        registry=_make_registry(),
        embed_fn=_make_embed_fn(),
    )

    decision = classifier.classify("I need a refund for my payment")
    assert decision.agent == "billing"
    assert 0.0 <= decision.confidence <= 1.0
    assert decision.reasoning is not None


def test_confidence_threshold_routes_to_default() -> None:
    def orthogonal_embed(texts: list[str]) -> list[list[float]]:
        """Embed fn where the query vector is orthogonal to all agent vectors."""
        vectors = []
        for text in texts:
            if "billing" in text.lower():
                vectors.append([1.0, 0.0, 0.0, 0.0])
            elif "technical" in text.lower():
                vectors.append([0.0, 1.0, 0.0, 0.0])
            elif "sales" in text.lower():
                vectors.append([0.0, 0.0, 1.0, 0.0])
            else:
                vectors.append([0.0, 0.0, 0.0, 1.0])
        return vectors

    registry = AgentRegistry()
    registry.register(
        AgentRegistration(name="billing", description="Handles billing issues.")
    )
    registry.register(
        AgentRegistration(name="technical", description="Handles technical errors.")
    )
    registry.register(
        AgentRegistration(name="general", description="Handles sales inquiries.")
    )

    classifier = FaissClassifier(
        registry=registry,
        embed_fn=orthogonal_embed,
        default_agent="general",
        confidence_threshold=0.5,
    )

    decision = classifier.classify("something completely unrelated")
    assert decision.agent == "general"
    assert "threshold" in (decision.reasoning or "").lower()


def test_classify_with_scores_returns_top_k() -> None:
    classifier = FaissClassifier(
        registry=_make_registry(),
        embed_fn=_make_embed_fn(),
    )

    results = classifier.classify_with_scores("billing issue", top_k=3)
    assert len(results) == 3
    assert results[0]["agent"] == "billing"
    assert all("similarity" in r for r in results)
    assert results[0]["similarity"] >= results[1]["similarity"]


def test_empty_registry_raises() -> None:
    registry = AgentRegistry()
    with pytest.raises(RegistryError):
        FaissClassifier(registry=registry, embed_fn=_make_embed_fn())


def test_non_routable_agents_excluded() -> None:
    registry = AgentRegistry()
    registry.register(
        AgentRegistration(name="billing", description="Handles billing.", routable=True)
    )
    registry.register(
        AgentRegistration(name="internal", description="Internal tool.", routable=False)
    )

    classifier = FaissClassifier(registry=registry, embed_fn=_make_embed_fn())
    decision = classifier.classify("anything at all")
    assert decision.agent != "internal"


def test_single_agent_always_matches() -> None:
    registry = AgentRegistry()
    registry.register(
        AgentRegistration(name="only_agent", description="The only agent.")
    )

    classifier = FaissClassifier(registry=registry, embed_fn=_make_embed_fn())
    decision = classifier.classify("literally anything")
    assert decision.agent == "only_agent"


def test_top_k_capped_to_agent_count() -> None:
    registry = AgentRegistry()
    registry.register(AgentRegistration(name="a", description="Agent A."))
    registry.register(AgentRegistration(name="b", description="Agent B."))

    classifier = FaissClassifier(registry=registry, embed_fn=_make_embed_fn())
    results = classifier.classify_with_scores("test", top_k=10)
    assert len(results) == 2
