from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout

from analytics.importance import collect_importance_stats, print_importance_analytics
from neuralstore import NeuralStore


class ImportanceAnalyticsTests(unittest.TestCase):
    def make_store(self) -> NeuralStore:
        return NeuralStore(seed=7, online_steps=1, replay_batch_size=2)

    def test_empty_scores_are_reported_without_statistics(self) -> None:
        stats = collect_importance_stats(self.make_store())
        self.assertEqual(stats["scored_records"], 0)
        self.assertIsNone(stats["mean"])
        self.assertEqual(stats["highest"], [])

        output = io.StringIO()
        with redirect_stdout(output):
            print_importance_analytics(self.make_store())
        self.assertIn("No valid importance scores", output.getvalue())
        self.assertIn("not objective measures", output.getvalue())

    def test_summary_ranks_valid_scores_and_excludes_invalid_values(self) -> None:
        store = self.make_store()
        store.record_importance = {
            "middle": 0.5,
            "high": 0.9,
            "low": 0.1,
            "nan": float("nan"),
            "outside": 1.5,
        }
        stats = collect_importance_stats(store)
        self.assertEqual(stats["scored_records"], 3)
        self.assertEqual(stats["mean"], 0.5)
        self.assertEqual(stats["median"], 0.5)
        self.assertEqual(stats["minimum"], 0.1)
        self.assertEqual(stats["maximum"], 0.9)
        self.assertEqual(stats["highest"][0], ("high", 0.9))
        self.assertEqual(stats["lowest"][0], ("low", 0.1))
        self.assertEqual(stats["invalid_scores"], ["nan", "outside"])

        output = io.StringIO()
        with redirect_stdout(output):
            print_importance_analytics(store)
        self.assertIn("reconstruction difficulty", output.getvalue())
        self.assertIn("Invalid scores excluded", output.getvalue())


if __name__ == "__main__":
    unittest.main()
