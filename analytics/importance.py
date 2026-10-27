"""Importance-score summaries for NeuralStore records."""

from __future__ import annotations

import math
from statistics import mean, median
from typing import Any

from neuralstore import NeuralStore


def collect_importance_stats(store: NeuralStore) -> dict[str, Any]:
    """Return summary statistics and ranked records with valid importance scores.

    Invalid, non-finite, or out-of-range scores are excluded from the summary
    and listed separately so corrupted/stale values are visible to the user.
    """
    valid: list[tuple[str, float]] = []
    invalid_ids: list[str] = []

    for record_id, raw_score in store.record_importance.items():
        try:
            score = float(raw_score)
        except (TypeError, ValueError):
            invalid_ids.append(record_id)
            continue
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            invalid_ids.append(record_id)
            continue
        valid.append((record_id, score))

    scores = [score for _, score in valid]
    ordered = sorted(valid, key=lambda item: (-item[1], item[0]))
    return {
        "scored_records": len(valid),
        "invalid_scores": invalid_ids,
        "mean": mean(scores) if scores else None,
        "median": median(scores) if scores else None,
        "minimum": min(scores) if scores else None,
        "maximum": max(scores) if scores else None,
        "highest": ordered[:5],
        "lowest": list(reversed(ordered[-5:])),
    }


def print_importance_analytics(store: NeuralStore) -> None:
    """Print importance-score distributions and top/bottom record IDs."""
    stats = collect_importance_stats(store)
    print("Importance analytics")
    print(f"Records with valid scores: {stats['scored_records']}")

    if stats["scored_records"] == 0:
        print("No valid importance scores are available.")
    else:
        print(f"Mean: {stats['mean']:.4f}")
        print(f"Median: {stats['median']:.4f}")
        print(f"Minimum: {stats['minimum']:.4f}")
        print(f"Maximum: {stats['maximum']:.4f}")
        print("Highest-scoring records:")
        for record_id, score in stats["highest"]:
            print(f"  {record_id}: {score:.4f}")
        print("Lowest-scoring records:")
        for record_id, score in stats["lowest"]:
            print(f"  {record_id}: {score:.4f}")

    if stats["invalid_scores"]:
        print("Invalid scores excluded (non-finite or outside [0, 1]):")
        for record_id in stats["invalid_scores"]:
            print(f"  {record_id}")

    print(
        "Interpretation: scores estimate reconstruction difficulty for the current "
        "model; they are not objective measures of business importance or usefulness."
    )
