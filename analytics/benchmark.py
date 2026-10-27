"""Read-only search latency benchmark for a loaded NeuralStore."""

from __future__ import annotations

import statistics
import time
from typing import Any

from neuralstore import NeuralStore


def benchmark_search(
    store: NeuralStore,
    iterations: int = 100,
    queries: list[Any] | None = None,
) -> dict[str, Any]:
    """Measure search latency without modifying records or model state."""
    if iterations < 1:
        raise ValueError("iterations must be >= 1")
    if not store.records:
        return {
            "iterations": 0,
            "records": 0,
            "median_ms": None,
            "p95_ms": None,
            "throughput_per_second": None,
        }

    if queries is None:
        queries = [next(iter(store.records.values()))]
    if not queries:
        raise ValueError("at least one query is required")

    durations: list[float] = []
    for index in range(iterations):
        query = queries[index % len(queries)]
        start = time.perf_counter()
        store.search(query, top_k=5)
        durations.append(time.perf_counter() - start)

    median_seconds = statistics.median(durations)
    ordered = sorted(durations)
    p95_index = max(0, min(len(ordered) - 1, int(0.95 * len(ordered) + 0.999999) - 1))
    p95_seconds = ordered[p95_index]
    total_seconds = sum(durations)
    return {
        "iterations": iterations,
        "records": len(store.records),
        "median_ms": median_seconds * 1000,
        "p95_ms": p95_seconds * 1000,
        "throughput_per_second": iterations / total_seconds if total_seconds else float("inf"),
    }


def print_search_benchmark(store: NeuralStore, iterations: int = 100) -> None:
    report = benchmark_search(store, iterations=iterations)
    print("Search benchmark (read-only; timings are local and approximate)")
    print(f"Records: {report['records']}")
    if report["iterations"] == 0:
        print("No records available; no searches were run.")
        return
    print(f"Search iterations: {report['iterations']}")
    print(f"Median latency: {report['median_ms']:.3f} ms")
    print(f"p95 latency: {report['p95_ms']:.3f} ms")
    print(f"Throughput: {report['throughput_per_second']:.2f} searches/sec")
