"""Labeled retrieval evaluation metrics for NeuralStore."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any

from neuralstore import NeuralStore


def load_labeled_queries(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSON list of {query, relevant_ids} evaluation cases."""
    source = Path(path)
    try:
        with source.open("r", encoding="utf-8") as file:
            payload = json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid evaluation JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc

    if not isinstance(payload, list) or not payload:
        raise ValueError("Evaluation file must contain a non-empty JSON list.")

    cases: list[dict[str, Any]] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict) or "query" not in item or "relevant_ids" not in item:
            raise ValueError(f"Evaluation entry {index} must contain query and relevant_ids.")
        relevant_ids = item["relevant_ids"]
        if (not isinstance(relevant_ids, list) or not relevant_ids
                or any(not isinstance(record_id, str) or not record_id for record_id in relevant_ids)):
            raise ValueError(f"relevant_ids in entry {index} must be a non-empty list of IDs.")
        if len(set(relevant_ids)) != len(relevant_ids):
            raise ValueError(f"relevant_ids in entry {index} contains duplicates.")
        cases.append({"query": item["query"], "relevant_ids": relevant_ids})
    return cases


def evaluate_retrieval(
    store: NeuralStore,
    labeled_queries: list[dict[str, Any]],
    top_k: int = 5,
    latent_weight: float = 0.8,
) -> dict[str, Any]:
    """Compute mean Precision@K, Recall@K, and reciprocal rank."""
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    if not 0.0 <= latent_weight <= 1.0:
        raise ValueError("latent_weight must be in [0, 1]")
    if not labeled_queries:
        raise ValueError("at least one labeled query is required")

    per_query = []
    for index, case in enumerate(labeled_queries):
        relevant_ids = set(case["relevant_ids"])
        results = store.search(case["query"], top_k=top_k, latent_weight=latent_weight)
        ranked_ids = [result["id"] for result in results]
        hits = [record_id for record_id in ranked_ids if record_id in relevant_ids]
        reciprocal_rank = next(
            (1.0 / rank for rank, record_id in enumerate(ranked_ids, start=1)
             if record_id in relevant_ids),
            0.0,
        )
        per_query.append({
            "query_index": index,
            "precision_at_k": len(hits) / top_k,
            "recall_at_k": len(hits) / len(relevant_ids),
            "reciprocal_rank": reciprocal_rank,
            "relevant_found": hits,
            "ranked_ids": ranked_ids,
        })

    return {
        "query_count": len(per_query),
        "top_k": top_k,
        "latent_weight": latent_weight,
        "precision_at_k": mean(item["precision_at_k"] for item in per_query),
        "recall_at_k": mean(item["recall_at_k"] for item in per_query),
        "mean_reciprocal_rank": mean(item["reciprocal_rank"] for item in per_query),
        "per_query": per_query,
    }


def print_retrieval_evaluation(
    store: NeuralStore,
    path: str | Path,
    top_k: int = 5,
    latent_weight: float = 0.8,
) -> None:
    """Load labeled query cases and print standard retrieval metrics."""
    cases = load_labeled_queries(path)
    report = evaluate_retrieval(store, cases, top_k=top_k, latent_weight=latent_weight)
    print("Labeled retrieval evaluation")
    print(f"Evaluation queries: {report['query_count']}")
    print(f"K: {report['top_k']}; latent weight: {report['latent_weight']:.2f}")
    print(f"Mean Precision@K: {report['precision_at_k']:.4f}")
    print(f"Mean Recall@K: {report['recall_at_k']:.4f}")
    print(f"Mean Reciprocal Rank: {report['mean_reciprocal_rank']:.4f}")
    print("Metrics are only as meaningful as the supplied relevance labels.")
