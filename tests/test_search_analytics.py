from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout

from analytics.search import collect_search_diagnostics, print_search_diagnostics
from neuralstore import NeuralStore


class SearchDiagnosticsTests(unittest.TestCase):
    def make_store(self) -> NeuralStore:
        return NeuralStore(seed=7, online_steps=1, replay_batch_size=2)

    def test_empty_store_reports_no_rankings(self) -> None:
        report = collect_search_diagnostics(self.make_store(), {"name": "Ada"})
        self.assertEqual(report["record_count"], 0)
        self.assertEqual(report["results"], [])
        self.assertEqual(report["rankings"], {})

        output = io.StringIO()
        with redirect_stdout(output):
            print_search_diagnostics(self.make_store(), {"name": "Ada"})
        self.assertIn("No records are stored", output.getvalue())

    def test_components_and_weighted_rankings_are_reported(self) -> None:
        store = self.make_store()
        store.insert_many([
            ("ada", {"name": "Ada", "role": "engineer"}),
            ("ada-copy", {"name": "Ada", "role": "engineer"}),
            ("linus", {"name": "Linus", "role": "maintainer"}),
        ])
        report = collect_search_diagnostics(
            store, {"name": "Ada", "role": "engineer"}, top_k=2
        )
        self.assertEqual(report["record_count"], 3)
        self.assertEqual(len(report["results"]), 2)
        for result in report["results"]:
            self.assertIn("latent_score", result)
            self.assertIn("feature_score", result)
            self.assertIn("score", result)
        self.assertEqual(set(report["rankings"]), {0.0, 0.25, 0.5, 0.75, 1.0})
        self.assertIn("ranking_overlap_with_default", report)
        self.assertTrue(report["near_duplicates"])

        output = io.StringIO()
        with redirect_stdout(output):
            print_search_diagnostics(store, {"name": "Ada", "role": "engineer"}, 2)
        self.assertIn("latent=", output.getvalue())
        self.assertIn("feature=", output.getvalue())
        self.assertIn("do not establish retrieval correctness", output.getvalue())

    def test_invalid_inputs_are_rejected(self) -> None:
        store = self.make_store()
        with self.assertRaises(ValueError):
            collect_search_diagnostics(store, {}, top_k=0)
        with self.assertRaises(ValueError):
            collect_search_diagnostics(store, {}, comparison_weights=(1.5,))


if __name__ == "__main__":
    unittest.main()
