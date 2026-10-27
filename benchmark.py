"""Draft microbenchmark for NeuralStore insert and search performance."""

from __future__ import annotations

import argparse
import statistics
import time
from typing import Any

from neuralstore import NeuralStore


def percentile(values: list[float], percent: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int((percent / 100) * len(ordered) + 0.5) - 1))
    return ordered[index]


def make_record(index: int) -> dict[str, Any]:
    return {
        "name": f"user-{index}",
        "department": f"team-{index % 10}",
        "active": index % 2 == 0,
        "score": index % 100,
        "profile": {
            "region": f"region-{index % 5}",
            "tags": [f"tag-{index % 7}", f"tag-{(index + 1) % 7}"],
        },
    }


def benchmark_size(size: int, online_steps: int, replay_batch_size: int,
                   search_repeats: int) -> None:
    store = NeuralStore(seed=7, online_steps=online_steps,
                        replay_batch_size=replay_batch_size)
    entries = [(f"record-{index}", make_record(index)) for index in range(size)]
    insert_times = []
    for record_id, record in entries:
        start = time.perf_counter()
        store.insert(record_id, record)
        insert_times.append(time.perf_counter() - start)

    bulk_store = NeuralStore(seed=7, online_steps=online_steps,
                             replay_batch_size=replay_batch_size)
    start = time.perf_counter()
    bulk_store.insert_many(entries)
    bulk_seconds = time.perf_counter() - start

    query = {"department": "team-3", "active": True}
    bulk_store.search(query, top_k=5)
    search_times = []
    for _ in range(search_repeats):
        start = time.perf_counter()
        bulk_store.search(query, top_k=5)
        search_times.append(time.perf_counter() - start)

    print(f"Records: {size:,}")
    print("  Insert median / p95: "
          f"{statistics.median(insert_times) * 1000:.2f} / "
          f"{percentile(insert_times, 95) * 1000:.2f} ms")
    print(f"  Bulk insert total / per record: {bulk_seconds * 1000:.2f} / "
          f"{bulk_seconds * 1000 / size:.2f} ms")
    print("  Search median / p95: "
          f"{statistics.median(search_times) * 1000:.2f} / "
          f"{percentile(search_times, 95) * 1000:.2f} ms")
    print(f"  Training steps (single / bulk): "
          f"{store.training_steps:,} / {bulk_store.training_steps:,}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Draft NeuralStore microbenchmark")
    parser.add_argument("--sizes", type=int, nargs="+", default=[100, 500, 1000])
    parser.add_argument("--online-steps", type=int, default=1)
    parser.add_argument("--replay-batch-size", type=int, default=8)
    parser.add_argument("--search-repeats", type=int, default=50)
    args = parser.parse_args()
    if any(size < 1 for size in args.sizes):
        parser.error("every --sizes value must be at least 1")
    if args.online_steps < 1 or args.replay_batch_size < 1:
        parser.error("--online-steps and --replay-batch-size must be at least 1")
    if args.search_repeats < 1:
        parser.error("--search-repeats must be at least 1")
    print("NeuralStore draft benchmark (timings are local and approximate)")
    for size in args.sizes:
        benchmark_size(size, args.online_steps, args.replay_batch_size,
                       args.search_repeats)


if __name__ == "__main__":
    main()
