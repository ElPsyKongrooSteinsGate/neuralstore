from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from analytics.benchmark import benchmark_search
from analytics.retrieval import evaluate_retrieval, load_labeled_queries
from analytics.training import collect_training_history
from neuralstore import NeuralStore


class RemainingAnalyticsTests(unittest.TestCase):
    def make_store(self) -> NeuralStore:
        return NeuralStore(seed=7, online_steps=1, replay_batch_size=2)

    def test_search_benchmark_runs_without_mutating_store(self) -> None:
        store = self.make_store()
        store.insert_many([("alpha", {"name": "Ada"}), ("beta", {"name": "Linus"})])
        records_before = dict(store.records)
        steps_before = store.training_steps
        report = benchmark_search(store, iterations=4, queries=[{"name": "Ada"}])
        self.assertEqual(report["iterations"], 4)
        self.assertEqual(report["records"], 2)
        self.assertGreaterEqual(report["median_ms"], 0)
        self.assertGreater(report["throughput_per_second"], 0)
        self.assertEqual(store.records, records_before)
        self.assertEqual(store.training_steps, steps_before)

    def test_training_history_is_recorded_and_persisted(self) -> None:
        store = self.make_store()
        self.assertEqual(collect_training_history(store)["points_reported"], 0)
        store.insert_many([("alpha", {"x": 1}), ("beta", {"x": 2})])
        history = collect_training_history(store, last=1)
        self.assertEqual(history["points_reported"], 1)
        self.assertEqual(history["last_loss"], store.last_loss)

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "history.pt"
            store.save(str(checkpoint))
            restored = NeuralStore.load(str(checkpoint))
            self.assertEqual(restored.loss_history, store.loss_history)
            self.assertEqual(restored.training_steps, store.training_steps)

    def test_labeled_retrieval_metrics_and_file_validation(self) -> None:
        store = self.make_store()
        store.insert_many([
            ("ada", {"name": "Ada", "role": "engineer"}),
            ("ada-copy", {"name": "Ada", "role": "engineer"}),
            ("linus", {"name": "Linus", "role": "maintainer"}),
        ])
        cases = [{"query": {"name": "Ada", "role": "engineer"},
                  "relevant_ids": ["ada", "ada-copy"]}]
        report = evaluate_retrieval(store, cases, top_k=2)
        self.assertEqual(report["query_count"], 1)
        self.assertGreaterEqual(report["precision_at_k"], 0.5)
        self.assertGreaterEqual(report["recall_at_k"], 0.5)
        self.assertGreater(report["mean_reciprocal_rank"], 0)

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "labels.json"
            source.write_text(json.dumps(cases), encoding="utf-8")
            self.assertEqual(load_labeled_queries(source), cases)
            source.write_text("{}", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_labeled_queries(source)


if __name__ == "__main__":
    unittest.main()
