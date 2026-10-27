"""Search ranking diagnostics for NeuralStore."""

from __future__ import annotations

from typing import Any

from neuralstore import NeuralStore


DEFAULT_LATENT_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
NEAR_DUPLICATE_FEATURE_THRESHOLD = 0.95


def collect_search_diagnostics(
    store: NeuralStore,
    query: Any,
    top_k: int = 5,
    comparison_weights: tuple[float, ...] = DEFAULT_LATENT_WEIGHTS,
) -> dict[str, Any]:
    """Return query match components and ranking comparisons across weights.

    This is descriptive only; it cannot determine whether a result is relevant
    without labeled examples or a separately defined relevance rule.
    """
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    if any(not 0.0 <= weight <= 1.0 for weight in comparison_weights):
        raise ValueError("comparison weights must be in [0, 1]")

    if not store.records:
        return {
            "record_count": 0,
            "top_k": top_k,
            "default_latent_weight": 0.8,
            "results": [],
            "rankings": {},
            "near_duplicates": [],
        }

    all_results = store.search(query, top_k=len(store.records), latent_weight=0.8)
    record_order = {record_id: index for index, record_id in enumerate(store.records)}

    results = all_results[:top_k]
    rankings: dict[float, list[str]] = {}
    for weight in comparison_weights:
        ranked = sorted(
            all_results,
            key=lambda result: (
                -(
                    weight * result["latent_score"]
                    + (1.0 - weight) * result["feature_score"]
                ),
                record_order[result["id"]],
            ),
        )
        rankings[weight] = [result["id"] for result in ranked[:top_k]]

    baseline = rankings.get(0.8)
    if baseline is None:
        baseline = [result["id"] for result in results]
    baseline_set = set(baseline)
    ranking_overlap = {
        weight: (len(baseline_set & set(ids)) / len(baseline_set) if baseline_set else 0.0)
        for weight, ids in rankings.items()
    }

    near_duplicates = [
        {
            "id": result["id"],
            "feature_score": result["feature_score"],
            "score": result["score"],
        }
        for result in all_results
        if result["feature_score"] >= NEAR_DUPLICATE_FEATURE_THRESHOLD
    ]

    return {
        "record_count": len(store.records),
        "top_k": top_k,
        "default_latent_weight": 0.8,
        "results": results,
        "rankings": rankings,
        "ranking_overlap_with_default": ranking_overlap,
        "near_duplicate_threshold": NEAR_DUPLICATE_FEATURE_THRESHOLD,
        "near_duplicates": near_duplicates,
    }


def print_search_diagnostics(store: NeuralStore, query: Any, top_k: int = 5) -> None:
    """Print per-component query scores and ranking changes by latent weight."""
    report = collect_search_diagnostics(store, query, top_k=top_k)
    print("Search diagnostics")
    print(f"Records scanned: {report['record_count']}")
    if not report["record_count"]:
        print("No records are stored; nothing to rank.")
        return

    print(f"Query: {query!r}")
    print(f"Top {min(top_k, report['record_count'])} results at latent weight 0.8:")
    for result in report["results"]:
        print(
            f"  {result['id']}: combined={result['score']:.4f}, "
            f"latent={result['latent_score']:.4f}, "
            f"features={result['feature_score']:.4f}, "
            f"importance={result['importance']:.4f}"
        )

    print("Ranking by latent_weight (top IDs; overlap with weight 0.8):")
    overlaps = report["ranking_overlap_with_default"]
    for weight, ids in report["rankings"].items():
        print(f"  {weight:.2f}: {', '.join(ids) or 'none'} "
              f"(overlap {overlaps[weight]:.0%})")

    print(
        "High feature-similarity candidates (feature score >= "
        f"{report['near_duplicate_threshold']:.2f}; heuristic only):"
    )
    if report["near_duplicates"]:
        for candidate in report["near_duplicates"][:top_k]:
            print(
                f"  {candidate['id']}: feature={candidate['feature_score']:.4f}, "
                f"combined={candidate['score']:.4f}"
            )
    else:
        print("  None")

    print(
        "Scores and ranking changes describe model behavior; they do not establish "
        "retrieval correctness or relevance without labeled examples."
    )
