"""Read-only consistency checks for NeuralStore's in-memory structures."""

from __future__ import annotations

import math
from typing import Any

import torch

from neuralstore import NeuralStore


def collect_integrity_report(store: NeuralStore) -> dict[str, Any]:
    """Check record-derived mappings, cached matrices, and importance values.

    This checks internal consistency only. It does not repair state, reload a
    checkpoint, or prove that search results are semantically correct.
    """
    issues: list[dict[str, str]] = []

    def issue(code: str, message: str) -> None:
        issues.append({"code": code, "message": message})

    record_ids = set(store.records)
    feature_ids = set(store.features)
    memory_ids = set(store.memory_keys)
    raw_key_ids = set(store.feature_keys)
    importance_ids = set(store.record_importance)

    for label, ids in (
        ("features", feature_ids),
        ("memory keys", memory_ids),
        ("feature keys", raw_key_ids),
    ):
        missing = sorted(record_ids - ids)
        extra = sorted(ids - record_ids)
        if missing:
            issue(f"missing_{label.replace(' ', '_')}",
                  f"Records missing {label}: {', '.join(missing)}")
        if extra:
            issue(f"orphaned_{label.replace(' ', '_')}",
                  f"{label.capitalize()} without records: {', '.join(extra)}")

    if importance_ids != record_ids:
        missing = sorted(record_ids - importance_ids)
        extra = sorted(importance_ids - record_ids)
        if missing:
            issue("missing_importance", f"Records missing importance scores: {', '.join(missing)}")
        if extra:
            issue("orphaned_importance", f"Importance scores without records: {', '.join(extra)}")

    expected_ids = list(store.records)
    if store._memory_ids != expected_ids:
        issue("memory_order_mismatch", "Cached memory ID order differs from record insertion order.")

    if store._memory_matrix.shape != (len(expected_ids), store.latent_dim):
        issue(
            "memory_matrix_shape",
            f"Latent matrix shape is {tuple(store._memory_matrix.shape)}; expected "
            f"({len(expected_ids)}, {store.latent_dim}).",
        )
    if store._feature_matrix.shape != (len(expected_ids), store.field_dim):
        issue(
            "feature_matrix_shape",
            f"Feature matrix shape is {tuple(store._feature_matrix.shape)}; expected "
            f"({len(expected_ids)}, {store.field_dim}).",
        )

    for record_id in sorted(record_ids & feature_ids):
        fields, mask = store.features[record_id]
        if fields.shape != (store.max_fields, store.field_dim):
            issue("feature_tensor_shape", f"Feature tensor for {record_id!r} has unexpected shape {tuple(fields.shape)}.")
        if mask.shape != (store.max_fields,):
            issue("feature_mask_shape", f"Feature mask for {record_id!r} has unexpected shape {tuple(mask.shape)}.")
        if not torch.isfinite(fields).all().item():
            issue("nonfinite_features", f"Feature tensor for {record_id!r} contains non-finite values.")
        if not torch.isfinite(mask).all().item() or not torch.all((mask == 0) | (mask == 1)).item():
            issue("invalid_feature_mask", f"Feature mask for {record_id!r} must contain only finite 0/1 values.")

    for record_id in sorted(record_ids & memory_ids):
        vector = store.memory_keys[record_id]
        if vector.shape != (store.latent_dim,):
            issue("memory_key_shape", f"Latent key for {record_id!r} has unexpected shape {tuple(vector.shape)}.")
        if not torch.isfinite(vector).all().item():
            issue("nonfinite_memory_key", f"Latent key for {record_id!r} contains non-finite values.")

    for record_id in sorted(record_ids & raw_key_ids):
        vector = store.feature_keys[record_id]
        if vector.shape != (store.field_dim,):
            issue("feature_key_shape", f"Feature key for {record_id!r} has unexpected shape {tuple(vector.shape)}.")
        if not torch.isfinite(vector).all().item():
            issue("nonfinite_feature_key", f"Feature key for {record_id!r} contains non-finite values.")

    for name, matrix in (("latent", store._memory_matrix), ("feature", store._feature_matrix)):
        if not torch.isfinite(matrix).all().item():
            issue(f"nonfinite_{name}_matrix", f"Cached {name} matrix contains non-finite values.")

    if store._memory_matrix.shape == (len(expected_ids), store.latent_dim):
        for index, record_id in enumerate(expected_ids):
            if record_id in store.memory_keys and not torch.allclose(
                store._memory_matrix[index], store.memory_keys[record_id]
            ):
                issue("latent_matrix_mismatch", f"Latent matrix row for {record_id!r} differs from its cached key.")
    if store._feature_matrix.shape == (len(expected_ids), store.field_dim):
        for index, record_id in enumerate(expected_ids):
            if record_id in store.feature_keys and not torch.allclose(
                store._feature_matrix[index], store.feature_keys[record_id]
            ):
                issue("feature_matrix_mismatch", f"Feature matrix row for {record_id!r} differs from its cached key.")

    for record_id, score in store.record_importance.items():
        try:
            numeric_score = float(score)
        except (TypeError, ValueError):
            issue("invalid_importance", f"Importance score for {record_id!r} is not numeric.")
            continue
        if not math.isfinite(numeric_score) or not 0.0 <= numeric_score <= 1.0:
            issue("invalid_importance", f"Importance score for {record_id!r} is non-finite or outside [0, 1].")

    for parameter_name, parameter in store.model.named_parameters():
        if not torch.isfinite(parameter).all().item():
            issue("nonfinite_model_parameter", f"Model parameter {parameter_name!r} contains non-finite values.")

    return {
        "ok": not issues,
        "records_checked": len(record_ids),
        "issues": issues,
        "issue_count": len(issues),
    }


def print_integrity_report(store: NeuralStore) -> None:
    """Print the results of a read-only in-memory integrity check."""
    report = collect_integrity_report(store)
    status = "PASS" if report["ok"] else "ISSUES FOUND"
    print(f"Integrity check: {status}")
    print(f"Records checked: {report['records_checked']}")
    print(f"Issues: {report['issue_count']}")
    if report["issues"]:
        for item in report["issues"]:
            print(f"  [{item['code']}] {item['message']}")
    print("This check is read-only and does not repair the store or validate search quality.")
