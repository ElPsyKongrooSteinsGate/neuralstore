"""Load records from JSON files into a NeuralStore instance."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from neuralstore import NeuralStore


def _records_from_payload(payload: Any) -> list[tuple[str, Any]]:
    """Normalize supported JSON shapes into (record_id, record) pairs.

    Supported formats are an ID-to-record mapping, a single object with ``id``
    and ``record`` fields, or a list of objects with those fields.
    """
    if isinstance(payload, dict) and "id" in payload and "record" in payload:
        items = [(payload["id"], payload["record"])]
    elif isinstance(payload, dict):
        items = list(payload.items())
    elif isinstance(payload, list):
        items = []
        for index, item in enumerate(payload):
            if not isinstance(item, dict) or "id" not in item or "record" not in item:
                raise ValueError(
                    f"Entry {index} must be an object with 'id' and 'record' fields."
                )
            items.append((item["id"], item["record"]))
    else:
        raise ValueError(
            "JSON root must be an ID-to-record object, a record object with "
            "'id' and 'record', or a list of such record objects."
        )

    normalized: list[tuple[str, Any]] = []
    seen_ids: set[str] = set()
    for record_id, record in items:
        if not isinstance(record_id, str) or not record_id.strip():
            raise ValueError("Each record ID must be a non-empty string.")
        if record_id in seen_ids:
            raise ValueError(f"Duplicate record ID in JSON file: {record_id!r}.")
        seen_ids.add(record_id)
        try:
            json.dumps(record, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Record {record_id!r} is not valid JSON data: {exc}") from exc
        normalized.append((record_id, record))

    return normalized


def import_json_file(store: NeuralStore, path: str | Path) -> int:
    """Insert records from a JSON file and return the number imported.

    The complete file is parsed and validated before insertion. Existing IDs
    are rejected rather than overwritten.
    """
    source = Path(path)
    try:
        with source.open("r", encoding="utf-8") as file:
            payload = json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON in {source} at line {exc.lineno}, column {exc.colno}: "
            f"{exc.msg}"
        ) from exc

    records = _records_from_payload(payload)
    conflicts = [record_id for record_id, _ in records if record_id in store.records]
    if conflicts:
        raise ValueError(
            "These record IDs already exist: "
            + ", ".join(repr(item) for item in conflicts)
        )

    return store.insert_many(records)
