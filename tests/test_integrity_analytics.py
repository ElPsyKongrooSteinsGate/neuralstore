from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout

import torch

from analytics.integrity import collect_integrity_report, print_integrity_report
from neuralstore import NeuralStore


class IntegrityAnalyticsTests(unittest.TestCase):
    def make_store(self) -> NeuralStore:
        return NeuralStore(seed=7, online_steps=1, replay_batch_size=2)

    def test_empty_store_passes_integrity_check(self) -> None:
        report = collect_integrity_report(self.make_store())
        self.assertTrue(report["ok"])
        self.assertEqual(report["records_checked"], 0)
        self.assertEqual(report["issues"], [])

    def test_consistent_populated_store_passes(self) -> None:
        store = self.make_store()
        store.insert_many([("alpha", {"x": 1}), ("beta", {"x": 2})])
        report = collect_integrity_report(store)
        self.assertTrue(report["ok"], report["issues"])
        self.assertEqual(report["records_checked"], 2)

    def test_report_detects_missing_orphan_and_invalid_state(self) -> None:
        store = self.make_store()
        store.insert("alpha", {"x": 1})
        store.memory_keys.pop("alpha")
        store.record_importance["alpha"] = float("nan")
        store.records["orphan"] = {"x": 2}
        store._memory_matrix = torch.empty(0, store.latent_dim)

        report = collect_integrity_report(store)
        codes = {item["code"] for item in report["issues"]}
        self.assertFalse(report["ok"])
        self.assertIn("missing_memory_keys", codes)
        self.assertIn("invalid_importance", codes)
        self.assertIn("missing_features", codes)
        self.assertIn("memory_matrix_shape", codes)

        output = io.StringIO()
        with redirect_stdout(output):
            print_integrity_report(store)
        self.assertIn("Integrity check: ISSUES FOUND", output.getvalue())
        self.assertIn("read-only", output.getvalue())


if __name__ == "__main__":
    unittest.main()
