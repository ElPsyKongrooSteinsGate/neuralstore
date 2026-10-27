from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from analytics.stats import collect_database_stats, print_database_stats
from neuralstore import NeuralStore


class DatabaseStatsTests(unittest.TestCase):
    def test_empty_store_reports_zero_sizes_and_empty_fields(self) -> None:
        store = NeuralStore(seed=7, online_steps=1, replay_batch_size=2)
        stats = collect_database_stats(store, "empty.pt")

        self.assertEqual(stats["records"], 0)
        self.assertEqual(stats["unique_ids"], 0)
        self.assertEqual(stats["serialized_record_bytes"], 0)
        self.assertEqual(stats["average_record_bytes"], 0)
        self.assertEqual(stats["median_record_bytes"], 0)
        self.assertFalse(stats["field_occurrences"])
        self.assertFalse(stats["value_types"])

    def test_populated_stats_count_nested_fields_types_and_empty_values(self) -> None:
        store = NeuralStore(seed=7, online_steps=1, replay_batch_size=2)
        store.insert_many([
            ("one", {
                "name": "Ada",
                "details": {"active": True, "note": ""},
                "items": [1, None],
                "nothing": {},
            }),
            ("two", {"name": "Linus", "score": 5}),
        ])

        stats = collect_database_stats(store, Path("example.pt"))
        self.assertEqual(stats["records"], 2)
        self.assertEqual(stats["unique_ids"], 2)
        self.assertGreater(stats["serialized_record_bytes"], 0)
        self.assertEqual(stats["field_occurrences"]["name"], 2)
        self.assertEqual(stats["field_occurrences"]["details.active"], 1)
        self.assertEqual(stats["value_types"]["bool"], 1)
        self.assertEqual(stats["value_types"]["int"], 2)
        self.assertEqual(stats["empty_values"]["null values"], 1)
        self.assertEqual(stats["empty_values"]["empty strings"], 1)
        self.assertEqual(stats["empty_values"]["empty objects"], 1)

        output = io.StringIO()
        with redirect_stdout(output):
            print_database_stats(store, "example.pt")
        report = output.getvalue()
        self.assertIn("Field occurrences (not record coverage)", report)
        self.assertIn("excludes checkpoint/model overhead", report)
        self.assertIn("Records: 2", report)


if __name__ == "__main__":
    unittest.main()
