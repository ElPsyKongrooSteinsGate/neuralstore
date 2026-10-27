from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from data.import_json import import_json_file
from neuralstore import NeuralStore


class NeuralStoreOptimizationTests(unittest.TestCase):
    def make_store(self) -> NeuralStore:
        return NeuralStore(seed=7, online_steps=1, replay_batch_size=2)

    def test_insert_many_preserves_records_and_refreshes_in_batches(self) -> None:
        store = self.make_store()
        entries = [(f"item-{index}", {"value": index}) for index in range(5)]

        self.assertEqual(store.insert_many(entries), len(entries))
        self.assertEqual(list(store.records), [record_id for record_id, _ in entries])
        self.assertEqual(store.training_steps, 3)  # ceil(5 / replay_batch_size)
        self.assertEqual(set(store.memory_keys), set(store.records))
        self.assertEqual(set(store.feature_keys), set(store.records))

    def test_insert_many_validates_all_ids_before_mutating(self) -> None:
        store = self.make_store()
        with self.assertRaises(KeyError):
            store.insert_many([("new", {"x": 1}), ("new", {"x": 2})])
        self.assertEqual(store.records, {})

    def test_vectorized_search_matches_scalar_score_order(self) -> None:
        store = self.make_store()
        store.insert_many(
            [(f"item-{index}", {"value": index, "group": index % 3}) for index in range(6)]
        )
        query = {"value": 3, "group": 0}
        q_fields, q_mask = store._make_features(query)
        q_latent = store._embed(q_fields, q_mask)
        q_raw = F.normalize((q_fields * q_mask.unsqueeze(-1)).sum(dim=0), p=2, dim=0)

        scalar_scores = {}
        for record_id in store.records:
            latent = torch.dot(q_latent, store.memory_keys[record_id]).item()
            feature = torch.dot(q_raw, store.feature_keys[record_id]).item()
            scalar_scores[record_id] = 0.8 * latent + 0.2 * feature

        expected = sorted(store.records, key=scalar_scores.get, reverse=True)
        actual = [result["id"] for result in store.search(query, top_k=6)]
        self.assertEqual(actual, expected)

    def test_update_delete_and_checkpoint_keep_indices_consistent(self) -> None:
        store = self.make_store()
        store.insert_many([("one", {"x": 1}), ("two", {"x": 2}), ("three", {"x": 3})])
        store.update("two", {"x": 22})
        self.assertEqual(list(store.records), ["one", "two", "three"])
        store.delete("one")
        self.assertEqual(set(store.memory_keys), set(store.records))
        self.assertEqual(set(store.feature_keys), set(store.records))

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "store.pt"
            store.save(str(checkpoint))
            restored = NeuralStore.load(str(checkpoint))
            self.assertEqual(restored.records, store.records)
            self.assertEqual(set(restored.feature_keys), set(restored.records))
            self.assertTrue(restored.search({"x": 22}))

    def test_json_import_uses_batch_insert(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "records.json"
            source.write_text(
                json.dumps([
                    {"id": "alpha", "record": {"x": 1}},
                    {"id": "beta", "record": {"x": 2}},
                ]),
                encoding="utf-8",
            )
            store = self.make_store()
            self.assertEqual(import_json_file(store, source), 2)
            self.assertEqual(list(store.records), ["alpha", "beta"])
            self.assertEqual(store.training_steps, 1)

    def test_sample_data_import_search_update_delete_and_reload(self) -> None:
        sample_path = Path(__file__).parent.parent / "data" / "samples" / "sampledata.json"
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "sample-store.pt"
            store = self.make_store()

            self.assertEqual(import_json_file(store, sample_path), 3)
            self.assertEqual(len(store.all_records()), 3)
            self.assertEqual(
                store.read("sampledata")["record_metadata"]["store_id"],
                "NYC-FLAGSHIP-04",
            )

            query = {"device_metadata": {"node_id": "EDGE-SW-IOT-09"}}
            matches = store.search(query, top_k=3)
            self.assertEqual(len(matches), 3)
            self.assertIn("telemetry-994", [match["id"] for match in matches])

            updated = dict(store.read("telemetry-994"))
            updated["test_update"] = True
            store.update("telemetry-994", updated)
            self.assertTrue(store.read("telemetry-994")["test_update"])

            store.save(str(checkpoint))
            loaded = NeuralStore.load(str(checkpoint))
            self.assertEqual(len(loaded.records), 3)
            self.assertTrue(loaded.read("telemetry-994")["test_update"])

            loaded.delete("supply-chain-logistics-8821")
            self.assertEqual(len(loaded.records), 2)
            self.assertNotIn("supply-chain-logistics-8821", loaded.records)
            loaded.save(str(checkpoint))
            reloaded = NeuralStore.load(str(checkpoint))
            self.assertEqual(len(reloaded.records), 2)
            self.assertTrue(reloaded.search(query, top_k=2))


if __name__ == "__main__":
    unittest.main()
