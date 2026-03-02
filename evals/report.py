"""Report generator: turns raw eval results into readable markdown.

Reads the JSON output from run_eval.py and produces:
- Summary table (accuracy, latency, cost per model)
- Breakdown by difficulty tier
- Per-scenario results
- Confusion matrix per model
- List of misclassifications

Usage:
    python evals/report.py evals/results/eval_20260224_123456.json
    python evals/report.py evals/results/eval_20260224_123456.json --output evals/results/report.md
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_results(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def group_by(results: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in results:
        groups[r[key]].append(r)
    return groups


def accuracy(results: list[dict[str, Any]]) -> float:
    if not results:
        return 0.0
    return sum(1 for r in results if r["correct"]) / len(results)


def avg_latency(results: list[dict[str, Any]]) -> float:
    latencies = [r["latency_ms"] for r in results if r["latency_ms"] is not None]
    return sum(latencies) / len(latencies) if latencies else 0.0


def total_cost(results: list[dict[str, Any]]) -> float:
    return sum(r["cost_usd"] for r in results if r["cost_usd"] is not None)


def cost_per_1k(results: list[dict[str, Any]]) -> float:
    n = len([r for r in results if r["cost_usd"] is not None])
    if n == 0:
        return 0.0
    return (total_cost(results) / n) * 1000


def generate_report(data: dict[str, Any]) -> str:
    metadata = data["metadata"]
    results = data["results"]
    lines: list[str] = []

    lines.append("# Eval Report: Classifier Prompt Benchmarking")
    lines.append("")
    lines.append(f"**Date:** {metadata['timestamp']}")
    lines.append(f"**Fixtures:** `{metadata['fixtures']}`")
    lines.append(f"**Models:** {', '.join(metadata['models'])}")
    lines.append(f"**Scenarios:** {', '.join(metadata['scenarios'])}")
    lines.append(f"**Total evaluations:** {metadata['total_cases']}")
    lines.append("")

    # --- Summary table ---
    lines.append("## Summary")
    lines.append("")
    lines.append("| Model | Accuracy | Avg Latency | Cost / 1k Routes | Total Cost |")
    lines.append("|-------|----------|-------------|-------------------|------------|")

    by_model = group_by(results, "model")
    for model, model_results in by_model.items():
        acc = accuracy(model_results)
        lat = avg_latency(model_results)
        c1k = cost_per_1k(model_results)
        tc = total_cost(model_results)
        lines.append(f"| {model} | {acc:.1%} | {lat:.0f}ms | ${c1k:.4f} | ${tc:.4f} |")

    lines.append("")

    # --- Breakdown by difficulty ---
    lines.append("## Accuracy by Difficulty")
    lines.append("")

    difficulties = ["easy", "hard", "adversarial"]
    header = "| Model |" + " | ".join(d.capitalize() for d in difficulties) + " |"
    sep = "|-------|" + " | ".join("---" for _ in difficulties) + " |"
    lines.append(header)
    lines.append(sep)

    for model, model_results in by_model.items():
        by_diff = group_by(model_results, "difficulty")
        cells = []
        for d in difficulties:
            d_results = by_diff.get(d, [])
            if d_results:
                acc = accuracy(d_results)
                cells.append(f"{acc:.1%} ({len(d_results)})")
            else:
                cells.append("—")
        lines.append(f"| {model} | " + " | ".join(cells) + " |")

    lines.append("")

    # --- Per-scenario results ---
    lines.append("## Per-Scenario Results")
    lines.append("")

    by_scenario = group_by(results, "scenario")
    for scenario, scenario_results in by_scenario.items():
        lines.append(f"### {scenario}")
        lines.append("")
        lines.append("| Model | Accuracy | Avg Latency | Cost / 1k |")
        lines.append("|-------|----------|-------------|-----------|")

        scenario_by_model = group_by(scenario_results, "model")
        for model, sr in scenario_by_model.items():
            acc = accuracy(sr)
            lat = avg_latency(sr)
            c1k = cost_per_1k(sr)
            lines.append(f"| {model} | {acc:.1%} | {lat:.0f}ms | ${c1k:.4f} |")

        lines.append("")

    # --- Confusion matrix per model ---
    lines.append("## Confusion Matrix")
    lines.append("")

    for model, model_results in by_model.items():
        lines.append(f"### {model}")
        lines.append("")
        lines.append("Rows = expected, Columns = predicted. Only misclassifications shown.")
        lines.append("")

        confusion: Counter[tuple[str, str]] = Counter()
        all_agents: set[str] = set()

        for r in model_results:
            if r["actual_agent"] is None:
                continue
            expected_list = r["expected_agent"]
            if isinstance(expected_list, str):
                expected_list = [expected_list]
            actual = r["actual_agent"].strip().lower()
            all_agents.add(actual)
            for e in expected_list:
                all_agents.add(e.strip().lower())
            if not r["correct"]:
                primary_expected = expected_list[0].strip().lower()
                confusion[(primary_expected, actual)] += 1

        if not confusion:
            lines.append("No misclassifications.")
            lines.append("")
            continue

        agents_sorted = sorted(all_agents)
        header = "| Expected \\ Predicted |" + " | ".join(agents_sorted) + " |"
        sep = "|---|" + " | ".join("---" for _ in agents_sorted) + " |"
        lines.append(header)
        lines.append(sep)

        for expected in agents_sorted:
            cells = []
            for predicted in agents_sorted:
                count = confusion.get((expected, predicted), 0)
                cells.append(str(count) if count > 0 else "·")
            lines.append(f"| {expected} | " + " | ".join(cells) + " |")

        lines.append("")

    # --- Misclassifications ---
    misses = [r for r in results if not r["correct"] and r["error"] is None]
    if misses:
        lines.append("## Misclassifications")
        lines.append("")
        lines.append("| Scenario | Case | Model | Message | Expected | Got | Confidence |")
        lines.append("|----------|------|-------|---------|----------|-----|------------|")

        for r in misses:
            expected = r["expected_agent"]
            if isinstance(expected, list):
                expected = ", ".join(expected)
            msg = r["message"][:60] + ("..." if len(r["message"]) > 60 else "")
            lines.append(
                f"| {r['scenario']} | {r['case_id']} | {r['model']} "
                f"| {msg} | {expected} | {r['actual_agent']} | {r['confidence']} |"
            )

        lines.append("")

    # --- Errors ---
    errors = [r for r in results if r["error"] is not None]
    if errors:
        lines.append("## Errors")
        lines.append("")
        for r in errors:
            lines.append(f"- **{r['case_id']}** ({r['model']}): {r['error']}")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate eval report from results.")
    parser.add_argument("results", type=Path, help="Path to eval results JSON file.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output markdown file (default: print to stdout).",
    )
    args = parser.parse_args()

    if not args.results.exists():
        print(f"Results file not found: {args.results}", file=sys.stderr)
        sys.exit(1)

    data = load_results(args.results)
    report = generate_report(data)

    if args.output:
        with open(args.output, "w") as f:
            f.write(report)
        print(f"Report saved to: {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()
