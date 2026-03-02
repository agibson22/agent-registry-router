"""Eval runner: benchmarks classifier prompt quality across LLMs.

Loads fixture scenarios, builds classifier prompts using the library,
calls each configured model, and records results to JSON.

Usage:
    python evals/run_eval.py
    python evals/run_eval.py --fixtures evals/fixtures.json --models gpt-4o-mini claude-haiku
    python evals/run_eval.py --scenarios customer_support education_tutors
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent_registry_router.core import AgentRegistry, build_classifier_system_prompt

ROOT_ENV = Path(__file__).parent.parent / ".env"


def _load_dotenv(path: Path = ROOT_ENV) -> None:
    """Load .env file into os.environ without adding a dependency."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        line = line.removeprefix("export ")
        key, _, value = line.partition("=")
        if key:
            os.environ.setdefault(key.strip(), value.strip())


EVALS_DIR = Path(__file__).parent
DEFAULT_FIXTURES = EVALS_DIR / "fixtures.json"
RESULTS_DIR = EVALS_DIR / "results"

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "agent": {
            "type": "string",
            "description": "The name of the agent to route to.",
        },
        "confidence": {
            "type": "number",
            "description": "Confidence in the routing decision, 0.0 to 1.0.",
        },
        "reasoning": {
            "type": "string",
            "description": "Short explanation for the routing choice.",
        },
    },
    "required": ["agent", "confidence", "reasoning"],
    "additionalProperties": False,
}

MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "gpt-4o-mini": {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "input_cost_per_1m": 0.15,
        "output_cost_per_1m": 0.60,
    },
    "claude-haiku": {
        "provider": "anthropic",
        "model": "claude-haiku-4-5-20251001",
        "input_cost_per_1m": 1.00,
        "output_cost_per_1m": 5.00,
    },
    "gemini-flash": {
        "provider": "google",
        "model": "gemini-2.5-flash",
        "input_cost_per_1m": 0.15,
        "output_cost_per_1m": 0.60,
    },
    "faiss-openai": {
        "provider": "faiss",
        "model": "text-embedding-3-small",
        "input_cost_per_1m": 0.02,
        "output_cost_per_1m": 0.0,
    },
}


def load_fixtures(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def classify_openai(
    system_prompt: str, user_message: str, model: str
) -> dict[str, Any]:
    from openai import OpenAI

    client = OpenAI()
    start = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "route_decision",
                "strict": True,
                "schema": RESPONSE_SCHEMA,
            },
        },
        temperature=0.0,
    )
    latency_ms = (time.perf_counter() - start) * 1000

    content = response.choices[0].message.content or "{}"
    usage = response.usage

    return {
        "decision": json.loads(content),
        "latency_ms": round(latency_ms, 1),
        "input_tokens": usage.prompt_tokens if usage else 0,
        "output_tokens": usage.completion_tokens if usage else 0,
    }


ANTHROPIC_ROUTE_TOOL = {
    "name": "route_decision",
    "description": "Submit the routing decision for the user's message.",
    "input_schema": RESPONSE_SCHEMA,
}


def classify_anthropic(
    system_prompt: str, user_message: str, model: str
) -> dict[str, Any]:
    from anthropic import Anthropic

    client = Anthropic()
    start = time.perf_counter()
    response = client.messages.create(  # type: ignore[call-overload]
        model=model,
        max_tokens=256,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
        tools=[ANTHROPIC_ROUTE_TOOL],
        tool_choice={"type": "tool", "name": "route_decision"},
        temperature=0.0,
    )
    latency_ms = (time.perf_counter() - start) * 1000

    decision = {"agent": "PARSE_ERROR", "confidence": 0.0, "reasoning": ""}
    for block in response.content:
        if block.type == "tool_use" and block.name == "route_decision":
            decision = block.input
            break

    return {
        "decision": decision,
        "latency_ms": round(latency_ms, 1),
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }


def classify_google(
    system_prompt: str, user_message: str, model: str
) -> dict[str, Any]:
    from google import genai
    from google.genai.types import GenerateContentConfig

    google_schema = {
        k: v for k, v in RESPONSE_SCHEMA.items() if k != "additionalProperties"
    }

    client = genai.Client()
    start = time.perf_counter()
    response = client.models.generate_content(
        model=model,
        contents=user_message,
        config=GenerateContentConfig(
            system_instruction=system_prompt,
            response_mime_type="application/json",
            response_json_schema=google_schema,
            temperature=0.0,
        ),
    )
    latency_ms = (time.perf_counter() - start) * 1000

    raw_text = response.text or "{}"
    try:
        decision = json.loads(raw_text)
    except json.JSONDecodeError:
        decision = {"agent": "PARSE_ERROR", "confidence": 0.0, "reasoning": raw_text}

    input_tokens = 0
    output_tokens = 0
    if response.usage_metadata:
        input_tokens = response.usage_metadata.prompt_token_count or 0
        output_tokens = response.usage_metadata.candidates_token_count or 0

    return {
        "decision": decision,
        "latency_ms": round(latency_ms, 1),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }


def _openai_embed(texts: list[str]) -> list[list[float]]:
    from openai import OpenAI

    client = OpenAI()
    response = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return [item.embedding for item in response.data]


_faiss_classifiers: dict[str, Any] = {}


def classify_faiss(
    system_prompt: str, user_message: str, model: str
) -> dict[str, Any]:
    """Classify using FAISS. system_prompt is ignored; classifier is cached per scenario."""
    from agent_registry_router.core.classifier import FaissClassifier

    scenario_key = system_prompt
    if scenario_key not in _faiss_classifiers:
        raise RuntimeError("FAISS classifier not initialized for this scenario.")

    classifier: FaissClassifier = _faiss_classifiers[scenario_key]
    start = time.perf_counter()
    decision = classifier.classify(user_message)
    latency_ms = (time.perf_counter() - start) * 1000

    return {
        "decision": {
            "agent": decision.agent,
            "confidence": decision.confidence,
            "reasoning": decision.reasoning,
        },
        "latency_ms": round(latency_ms, 1),
        "input_tokens": 0,
        "output_tokens": 0,
    }


PROVIDER_CLASSIFIERS = {
    "openai": classify_openai,
    "anthropic": classify_anthropic,
    "google": classify_google,
    "faiss": classify_faiss,
}


def is_correct(expected: str | list[str], actual: str) -> bool:
    actual_normalized = actual.strip().lower()
    if isinstance(expected, list):
        return actual_normalized in [e.strip().lower() for e in expected]
    return actual_normalized == expected.strip().lower()


def run_eval(
    fixtures_path: Path,
    model_keys: list[str],
    scenario_ids: list[str] | None = None,
) -> dict[str, Any]:
    fixtures = load_fixtures(fixtures_path)
    scenarios = fixtures["scenarios"]
    if scenario_ids:
        scenarios = [s for s in scenarios if s["id"] in scenario_ids]

    results: list[dict[str, Any]] = []
    total_cases = sum(len(s["cases"]) for s in scenarios) * len(model_keys)
    completed = 0

    for scenario in scenarios:
        registry = AgentRegistry.from_descriptions(scenario["agents"])
        system_prompt = build_classifier_system_prompt(
            registry, default_agent=scenario["default_agent"]
        )

        faiss_initialized = False
        for model_key in model_keys:
            config = MODEL_CONFIGS[model_key]
            classify_fn = PROVIDER_CLASSIFIERS[config["provider"]]

            if config["provider"] == "faiss" and not faiss_initialized:
                from agent_registry_router.core.classifier import FaissClassifier

                _faiss_classifiers[system_prompt] = FaissClassifier(
                    registry=registry,
                    embed_fn=_openai_embed,
                    default_agent=scenario["default_agent"],
                )
                faiss_initialized = True

            print(f"\n--- {scenario['id']} / {model_key} ---")

            for case in scenario["cases"]:
                completed += 1
                try:
                    response = classify_fn(
                        system_prompt, case["message"], config["model"]
                    )
                    decision = response["decision"]
                    agent_picked = decision.get("agent", "MISSING")
                    correct = is_correct(case["expected_agent"], agent_picked)

                    input_cost = (response["input_tokens"] / 1_000_000) * config[
                        "input_cost_per_1m"
                    ]
                    output_cost = (response["output_tokens"] / 1_000_000) * config[
                        "output_cost_per_1m"
                    ]

                    result = {
                        "scenario": scenario["id"],
                        "case_id": case["id"],
                        "model": model_key,
                        "message": case["message"],
                        "expected_agent": case["expected_agent"],
                        "actual_agent": agent_picked,
                        "correct": correct,
                        "confidence": decision.get("confidence", 0.0),
                        "reasoning": decision.get("reasoning", ""),
                        "difficulty": case["difficulty"],
                        "latency_ms": response["latency_ms"],
                        "input_tokens": response["input_tokens"],
                        "output_tokens": response["output_tokens"],
                        "cost_usd": round(input_cost + output_cost, 6),
                        "error": None,
                    }

                    status = "✓" if correct else "✗"
                    print(
                        f"  [{completed}/{total_cases}] {status} {case['id']}: "
                        f"expected={case['expected_agent']}, "
                        f"got={agent_picked} ({decision.get('confidence', '?')})"
                    )

                except Exception as e:
                    result = {
                        "scenario": scenario["id"],
                        "case_id": case["id"],
                        "model": model_key,
                        "message": case["message"],
                        "expected_agent": case["expected_agent"],
                        "actual_agent": None,
                        "correct": False,
                        "confidence": None,
                        "reasoning": None,
                        "difficulty": case["difficulty"],
                        "latency_ms": None,
                        "input_tokens": None,
                        "output_tokens": None,
                        "cost_usd": None,
                        "error": str(e),
                    }
                    print(f"  [{completed}/{total_cases}] ⚠ {case['id']}: ERROR {e}")

                results.append(result)

    run_metadata = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "fixtures": str(fixtures_path),
        "models": model_keys,
        "scenarios": [s["id"] for s in scenarios],
        "total_cases": total_cases,
    }

    return {"metadata": run_metadata, "results": results}


def main() -> None:
    _load_dotenv()
    parser = argparse.ArgumentParser(description="Run classifier prompt eval suite.")
    parser.add_argument(
        "--fixtures",
        type=Path,
        default=DEFAULT_FIXTURES,
        help="Path to fixtures JSON file.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODEL_CONFIGS.keys()),
        choices=list(MODEL_CONFIGS.keys()),
        help="Models to evaluate.",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=None,
        help="Scenario IDs to run (default: all).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file path (default: evals/results/eval_<timestamp>.json).",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        default=False,
        help="Auto-generate markdown report after eval.",
    )
    args = parser.parse_args()

    if not args.fixtures.exists():
        print(f"Fixtures file not found: {args.fixtures}", file=sys.stderr)
        sys.exit(1)

    print(f"Fixtures: {args.fixtures}")
    print(f"Models:   {args.models}")
    print(f"Scenarios: {args.scenarios or 'all'}")

    output = run_eval(args.fixtures, args.models, args.scenarios)

    if args.output:
        out_path = args.output
    else:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_path = RESULTS_DIR / f"eval_{ts}.json"

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    correct = sum(1 for r in output["results"] if r["correct"])
    total = len(output["results"])
    print(f"\n{'=' * 50}")
    print(f"Results: {correct}/{total} correct ({correct / total * 100:.1f}%)")
    print(f"Saved to: {out_path}")

    if args.report:
        import importlib.util

        spec = importlib.util.spec_from_file_location("report", EVALS_DIR / "report.py")
        report_mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(report_mod)  # type: ignore[union-attr]

        report_path = RESULTS_DIR / "report.md"
        report = report_mod.generate_report(output)
        with open(report_path, "w") as f:
            f.write(report)
        print(f"Report:  {report_path}")


if __name__ == "__main__":
    main()
