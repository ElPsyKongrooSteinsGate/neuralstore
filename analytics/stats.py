"""Database and model statistics for NeuralStore."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any

from neuralstore import NeuralStore


def _walk_record(
    value: Any,
    field_counts: Counter[str],
    type_counts: Counter[str],
    empty_counts: Counter[str],
    prefix: str = "",
) -> None:
    """Collect field occurrences, leaf types, and empty-value counts."""
    if isinstance(value, dict):
        if not value:
            empty_counts["empty objects"] += 1
        for key, child in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            field_counts[name] += 1
            _walk_record(child, field_counts, type_counts, empty_counts, name)
    elif isinstance(value, list):
        if not value:
            empty_counts["empty arrays"] += 1
        for child in value:
            _walk_record(child, field_counts, type_counts, empty_counts, f"{prefix}[]")
    else:
        type_counts[type(value).__name__] += 1
        if value is None:
            empty_counts["null values"] += 1
        elif isinstance(value, str) and not value.strip():
            empty_counts["empty strings"] += 1


def collect_database_stats(store: NeuralStore, db_path: str | Path) -> dict[str, Any]:
    """Return database, record-size, field, type, and model statistics."""
    field_counts: Counter[str] = Counter()
    type_counts: Counter[str] = Counter()
    empty_counts: Counter[str] = Counter()
    record_sizes: list[int] = []

    for record in store.records.values():
        serialized = json.dumps(record, ensure_ascii=False, allow_nan=False).encode("utf-8")
        record_sizes.append(len(serialized))
        _walk_record(record, field_counts, type_counts, empty_counts)

    count = len(store.records)
    return {
        "database": str(db_path),
        "records": count,
        "unique_ids": len(set(store.records)),
        "training_steps": store.training_steps,
        "last_loss": store.last_loss,
        "latent_dimensions": store.latent_dim,
        "serialized_record_bytes": sum(record_sizes),
        "average_record_bytes": mean(record_sizes) if record_sizes else 0.0,
        "median_record_bytes": median(record_sizes) if record_sizes else 0.0,
        "field_occurrences": field_counts,
        "value_types": type_counts,
        "empty_values": empty_counts,
    }


def print_database_stats(store: NeuralStore, db_path: str | Path) -> None:
    """Print a readable stats report. Record sizes exclude checkpoint overhead."""
    stats = collect_database_stats(store, db_path)
    print(f"Database: {stats['database']}")
    print(f"Records: {stats['records']}")
    print(f"Unique IDs: {stats['unique_ids']}")
    print(f"Training steps: {stats['training_steps']}")
    print(f"Last training loss: {stats['last_loss']}")
    print(f"Latent dimensions: {stats['latent_dimensions']}")
    print("Serialized record payload bytes (excludes checkpoint/model overhead): "
          f"{stats['serialized_record_bytes']}")
    print(f"Average record payload bytes: {stats['average_record_bytes']:.1f}")
    print(f"Median record payload bytes: {stats['median_record_bytes']:.1f}")

    field_counts: Counter[str] = stats["field_occurrences"]
    print("Field occurrences (not record coverage):")
    if field_counts:
        for name, occurrences in field_counts.most_common(10):
            print(f"  {name}: {occurrences}")
    else:
        print("  None")

    type_counts: Counter[str] = stats["value_types"]
    print("Leaf value types:")
    if type_counts:
        for name, occurrences in sorted(type_counts.items()):
            print(f"  {name}: {occurrences}")
    else:
        print("  None")

    empty_counts: Counter[str] = stats["empty_values"]
    print("Null/empty values:")
    if empty_counts:
        for name, occurrences in sorted(empty_counts.items()):
            print(f"  {name}: {occurrences}")
    else:
        print("  None")
