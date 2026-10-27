from __future__ import annotations

import hashlib
import json
import math
import random
import re
from typing import Any, Optional

import torch
from torch import nn
import torch.nn.functional as F


# ============================================================
# 1. Deterministic mixed-data feature encoder
# ============================================================

class MixedFeatureEncoder:
    """
    Converts JSON-like records into a set of field vectors.

    Supports strings, numbers, booleans, lists and nested dictionaries.
    No learned parameters here: learning happens in NeuralModel.
    """

    def __init__(self, field_dim: int = 64, max_fields: int = 32):
        if field_dim < 8 or max_fields < 1:
            raise ValueError("field_dim must be >= 8 and max_fields >= 1")
        self.field_dim = field_dim
        self.max_fields = max_fields

    def _hash(self, text: str) -> tuple[int, float]:
        digest = hashlib.blake2b(
            text.encode("utf-8"), digest_size=8
        ).digest()
        n = int.from_bytes(digest, "little")
        return n % self.field_dim, (1.0 if (n >> 32) & 1 else -1.0)

    def _add(self, vector: torch.Tensor, token: str, value=1.0):
        index, sign = self._hash(token)
        vector[index] += sign * float(value)

    def _flatten(self, record: Any) -> list[tuple[str, Any]]:
        fields: list[tuple[str, Any]] = []

        def visit(value: Any, path: str):
            if isinstance(value, dict):
                if not value:
                    fields.append((path, "<empty-object>"))
                for key in sorted(value, key=str):
                    visit(value[key], f"{path}.{key}" if path else str(key))
            elif isinstance(value, (list, tuple)):
                if not value:
                    fields.append((path, "<empty-list>"))
                for item in value:
                    visit(item, path + "[]")
            else:
                fields.append((path or "value", value))

        visit(record, "")
        fields.sort(key=lambda item: (item[0], str(item[1])))
        return fields[: self.max_fields]

    def _field_vector(self, name: str, value: Any) -> torch.Tensor:
        v = torch.zeros(self.field_dim, dtype=torch.float32)

        self._add(v, f"field:{name}")

        if value is None:
            self._add(v, f"{name}:null")
        elif isinstance(value, bool):
            self._add(v, f"{name}:bool:{value}")
        elif isinstance(value, (int, float)):
            number = float(value)
            if math.isfinite(number):
                self._add(v, f"{name}:numeric-present")
                self._add(v, f"{name}:numeric", math.tanh(number / 100.0))
            else:
                self._add(v, f"{name}:nonfinite")
        elif isinstance(value, str):
            text = value.strip().lower()
            self._add(v, f"{name}:category:{text}")

            tokens = re.findall(r"\w+", text, flags=re.UNICODE)
            for token in tokens:
                self._add(v, f"{name}:token:{token}")
            for a, b in zip(tokens, tokens[1:]):
                self._add(v, f"{name}:bigram:{a}_{b}")
        else:
            self._add(v, f"{name}:type:{type(value).__name__}")
            self._add(v, f"{name}:value:{str(value)}")

        return F.normalize(v, p=2, dim=0)

    def transform(self, record: Any) -> tuple[torch.Tensor, torch.Tensor]:
        fields = self._flatten(record)

        vectors = torch.zeros(
            self.max_fields, self.field_dim, dtype=torch.float32
        )
        mask = torch.zeros(self.max_fields, dtype=torch.float32)

        for i, (name, value) in enumerate(fields):
            vectors[i] = self._field_vector(name, value)
            mask[i] = 1.0

        return vectors, mask


# ============================================================
# 2. Neural Model with Gated Pooling & Trained Importance
# ============================================================

class NeuralModel(nn.Module):
    """
    Gated field pooling, autoencoder reconstruction, next predictor,
    and a trained importance-weighted recurrent state module.
    """

    def __init__(
        self,
        field_dim: int = 64,
        hidden_dim: int = 96,
        latent_dim: int = 32,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        # Shared per-field transformation & gating
        self.field_projection = nn.Sequential(
            nn.Linear(field_dim, hidden_dim),
            nn.ReLU(),
        )
        self.field_gate = nn.Linear(field_dim, hidden_dim)

        self.to_latent = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        # Reconstruct aggregate field representation
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, field_dim),
        )

        # Importance Estimator Head (Scores record persistence salience: 0 to 1)
        self.importance_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        # Recurrent state-space module
        self.state_input = nn.Linear(latent_dim, latent_dim)
        self.state_hidden = nn.Linear(latent_dim, latent_dim, bias=False)
        self.state_gate = nn.Linear(2 * latent_dim, latent_dim)

        # Predict next record's latent representation
        self.next_predictor = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def encode(
        self, fields: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        h = self.field_projection(fields)
        gates = torch.sigmoid(self.field_gate(fields))

        weights = mask.unsqueeze(-1)
        pooled = (h * gates * weights).sum(dim=1)
        pooled = pooled / weights.sum(dim=1).clamp_min(1.0)

        z = self.to_latent(pooled)
        return F.normalize(z, p=2, dim=-1)

    def forward(
        self, fields: torch.Tensor, mask: torch.Tensor
    ):
        z = self.encode(fields, mask)

        weights = mask.unsqueeze(-1)
        target = (fields * weights).sum(dim=1)
        target = target / weights.sum(dim=1).clamp_min(1.0)

        reconstructed = self.decoder(z)
        return z, reconstructed, target

    def advance_state_with_importance(
        self, old_state: torch.Tensor, current_z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Updates memory state while factoring in the dynamically trained importance score.
        High importance modulates fading and preserves long-term persistence.
        """
        if old_state.dim() == 1:
            old_state = old_state.unsqueeze(0)
        if current_z.dim() == 1:
            current_z = current_z.unsqueeze(0)

        importance = self.importance_head(current_z) # [batch, 1]

        candidate = torch.tanh(
            self.state_input(current_z) + self.state_hidden(old_state)
        )
        base_gate = torch.sigmoid(
            self.state_gate(torch.cat([old_state, current_z], dim=-1))
        )

        # High importance should move the retention gate toward 1.
        effective_gate = base_gate + importance * (1.0 - base_gate)
        new_state = effective_gate * old_state + (1.0 - effective_gate) * candidate

        return new_state.squeeze(0), importance.squeeze(0)

    def predict_next(self, state: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.next_predictor(state), p=2, dim=-1)


# ============================================================
# 3. NeuralStore: Exact Records + Trained Importance Memory
# ============================================================

class NeuralStore:
    """
    Educational hybrid neural database featuring:
    - Exact JSON dictionary storage
    - Gated field pooling & representation learning
    - Surprise-trained importance-weighted recurrent stream memory
    - Online learning with experience replay
    """

    VERSION = 5

    def __init__(
        self,
        field_dim: int = 64,
        max_fields: int = 32,
        hidden_dim: int = 96,
        latent_dim: int = 32,
        learning_rate: float = 1e-3,
        replay_batch_size: int = 16,
        online_steps: int = 3,
        seed: int = 7,
        autosave_path: Optional[str] = None,
    ):
        if replay_batch_size < 1 or online_steps < 1:
            raise ValueError("replay_batch_size and online_steps must be >= 1")

        random.seed(seed)
        torch.manual_seed(seed)

        self.field_dim = field_dim
        self.max_fields = max_fields
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.replay_batch_size = replay_batch_size
        self.online_steps = online_steps
        # If set, persist after each successful insert/update/delete.
        self.autosave_path = autosave_path

        self.featurizer = MixedFeatureEncoder(field_dim, max_fields)
        self.model = NeuralModel(field_dim, hidden_dim, latent_dim)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=learning_rate
        )

        self.records: dict[str, Any] = {}
        self.features: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self.memory_keys: dict[str, torch.Tensor] = {}
        self.feature_keys: dict[str, torch.Tensor] = {}
        self._memory_ids: list[str] = []
        self._memory_matrix = torch.empty(0, latent_dim)
        self._feature_matrix = torch.empty(0, field_dim)
        self.record_importance: dict[str, float] = {}

        self.stream_state = torch.zeros(latent_dim)
        # State immediately before the most recently processed record. Together
        # with last_latent, this supports training on adjacent record pairs.
        self.last_state_before: Optional[torch.Tensor] = None
        self.last_latent: Optional[torch.Tensor] = None

        self.training_steps = 0
        self.last_loss: Optional[float] = None
        self.model.eval()

    @staticmethod
    def _check_id(record_id: str):
        if not isinstance(record_id, str) or not record_id:
            raise ValueError("record_id must be a non-empty string")

    @staticmethod
    def _check_json(record: Any):
        json.dumps(record, allow_nan=False)

    def _make_features(
        self, record: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.featurizer.transform(record)

    def _embed(
        self, fields: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            return self.model.encode(
                fields.unsqueeze(0), mask.unsqueeze(0)
            ).squeeze(0).clone()

    def _refresh_memory(self):
        """Recompute latent and raw-feature keys in bounded batches."""
        self.model.eval()
        memory_keys: dict[str, torch.Tensor] = {}
        feature_keys: dict[str, torch.Tensor] = {}
        ids = list(self.features)

        with torch.no_grad():
            for start in range(0, len(ids), 256):
                batch_ids = ids[start : start + 256]
                fields = torch.stack([self.features[rid][0] for rid in batch_ids])
                masks = torch.stack([self.features[rid][1] for rid in batch_ids])
                embeddings = self.model.encode(fields, masks)

                raw = (fields * masks.unsqueeze(-1)).sum(dim=1)
                raw = F.normalize(raw, p=2, dim=-1)
                for index, rid in enumerate(batch_ids):
                    memory_keys[rid] = embeddings[index].clone()
                    feature_keys[rid] = raw[index].clone()

        self.memory_keys = memory_keys
        self.feature_keys = feature_keys
        self._memory_ids = ids
        if ids:
            self._memory_matrix = torch.stack([memory_keys[rid] for rid in ids])
            self._feature_matrix = torch.stack([feature_keys[rid] for rid in ids])
        else:
            self._memory_matrix = torch.empty(0, self.latent_dim)
            self._feature_matrix = torch.empty(0, self.field_dim)

    def _learn_online(
        self,
        new_id: str,
        refresh_memory: bool = True,
        train_sequence: bool = True,
        advance_stream: bool = True,
    ):
        ids = list(self.features)
        if not ids:
            return

        old_state = self.stream_state.clone()
        for _ in range(self.online_steps):
            count = min(self.replay_batch_size, len(ids))
            sampled = random.sample(ids, count)

            if new_id not in sampled:
                sampled[-1] = new_id

            field_batch = torch.stack(
                [self.features[rid][0] for rid in sampled]
            )
            mask_batch = torch.stack(
                [self.features[rid][1] for rid in sampled]
            )

            current_fields, current_mask = self.features[new_id]
            current_fields_unsqueezed = current_fields.unsqueeze(0)
            current_mask_unsqueezed = current_mask.unsqueeze(0)

            self.model.train()

            # 1. Reconstruction loss over the replay batch
            _, reconstructed, target = self.model(field_batch, mask_batch)
            reconstruction_loss = F.mse_loss(reconstructed, target)

            # 2. Sequence prediction loss. The transition receives the previous
            # record latent and its output predicts the current record latent.
            # This gives state_input/state_hidden/state_gate real gradients.
            current_z, current_reconstructed, current_target = self.model(
                current_fields_unsqueezed, current_mask_unsqueezed
            )
            if train_sequence and self.last_latent is not None:
                previous_z = self.last_latent.detach().unsqueeze(0)
                transitioned_state, _ = self.model.advance_state_with_importance(
                    old_state.unsqueeze(0), previous_z
                )
                predicted_next = self.model.predict_next(transitioned_state.unsqueeze(0))
                prediction_loss = F.mse_loss(predicted_next, current_z.detach())
            else:
                # No adjacent pair exists on the first insertion.
                prediction_loss = current_z.new_zeros(())

            # 3. Surprise-Driven Importance Learning (Trained via Reconstruction Error)
            item_reconstruction_error = torch.mean(
                (current_reconstructed - current_target) ** 2, dim=-1
            )
            target_importance = torch.clamp(
                item_reconstruction_error * 10.0, 0.0, 1.0
            ).detach()

            importance_pred = self.model.importance_head(current_z)
            importance_loss = F.mse_loss(
                importance_pred.squeeze(-1), target_importance
            )

            # Total combined loss ensuring gradient flow into the importance head
            loss = (
                reconstruction_loss
                + 0.1 * prediction_loss
                + 0.2 * importance_loss
            )

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=1.0
            )
            self.optimizer.step()

            self.training_steps += 1
            self.last_loss = float(loss.detach().item())

        self.model.eval()
        with torch.no_grad():
            current_z = self.model.encode(
                current_fields_unsqueezed, current_mask_unsqueezed
            ).squeeze(0)
            importance = self.model.importance_head(current_z.unsqueeze(0)).squeeze()

            if advance_stream:
                new_state, importance = self.model.advance_state_with_importance(
                    old_state, current_z
                )
                self.stream_state = new_state.detach().clone()
                self.last_state_before = old_state.detach().clone()
                self.last_latent = current_z.detach().clone()
            self.record_importance[new_id] = float(importance.item())

        self.model.eval()
        if refresh_memory:
            self._refresh_memory()

    def _learn_batch(self, new_ids: list[str]) -> None:
        """Train new records in mini-batches and advance stream state in order."""
        historical_ids = [rid for rid in self.records if rid not in set(new_ids)]

        for start in range(0, len(new_ids), self.replay_batch_size):
            batch_ids = new_ids[start : start + self.replay_batch_size]
            replay_count = max(0, self.replay_batch_size - len(batch_ids))
            replay_ids = random.sample(
                historical_ids, min(replay_count, len(historical_ids))
            )
            fields = torch.stack(
                [self.features[rid][0] for rid in replay_ids + batch_ids]
            )
            masks = torch.stack(
                [self.features[rid][1] for rid in replay_ids + batch_ids]
            )
            current_count = len(batch_ids)

            for _ in range(self.online_steps):
                self.model.train()
                z, reconstructed, target = self.model(fields, masks)
                reconstruction_loss = F.mse_loss(reconstructed, target)

                current_z = z[:current_count]
                current_reconstruction_error = torch.mean(
                    (reconstructed[:current_count] - target[:current_count]) ** 2,
                    dim=-1,
                )
                target_importance = torch.clamp(
                    current_reconstruction_error * 10.0, 0.0, 1.0
                ).detach()
                predicted_importance = self.model.importance_head(current_z).squeeze(-1)
                importance_loss = F.mse_loss(predicted_importance, target_importance)

                if self.last_latent is not None:
                    transitioned_state, _ = self.model.advance_state_with_importance(
                        self.stream_state, self.last_latent
                    )
                    predicted_next = self.model.predict_next(
                        transitioned_state.unsqueeze(0)
                    )
                    prediction_loss = F.mse_loss(
                        predicted_next, current_z[:1].detach()
                    )
                else:
                    prediction_loss = reconstruction_loss.new_zeros(())

                loss = (
                    reconstruction_loss
                    + 0.1 * prediction_loss
                    + 0.2 * importance_loss
                )
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=1.0
                )
                self.optimizer.step()
                self.training_steps += 1
                self.last_loss = float(loss.detach().item())

            self.model.eval()
            with torch.no_grad():
                current_fields = fields[:current_count]
                current_masks = masks[:current_count]
                current_latents = self.model.encode(current_fields, current_masks)
                state = self.stream_state
                for index, record_id in enumerate(batch_ids):
                    state_before = state.clone()
                    state, importance = self.model.advance_state_with_importance(
                        state, current_latents[index]
                    )
                    state = state.detach().clone()
                    self.last_state_before = state_before
                    self.last_latent = current_latents[index].detach().clone()
                    self.record_importance[record_id] = float(importance.item())
                self.stream_state = state

    # ---------------- Exact CRUD ----------------

    def insert(self, record_id: str, record: Any):
        self.insert_many([(record_id, record)])

    def insert_many(self, records: list[tuple[str, Any]]) -> int:
        """Insert multiple records, refreshing all learned keys once.

        IDs and JSON values are validated before any records are added. Records
        are trained sequentially in the provided order, preserving stream order.
        """
        if not records:
            return 0

        seen_ids: set[str] = set()
        prepared: list[tuple[str, Any, tuple[torch.Tensor, torch.Tensor]]] = []
        for record_id, record in records:
            self._check_id(record_id)
            self._check_json(record)
            if record_id in seen_ids or record_id in self.records:
                raise KeyError(f"Record already exists: {record_id}")
            seen_ids.add(record_id)
            prepared.append((record_id, record, self._make_features(record)))

        new_ids = [record_id for record_id, _, _ in prepared]
        for record_id, record, features in prepared:
            self.records[record_id] = record
            self.features[record_id] = features

        if len(prepared) == 1:
            self._learn_online(new_ids[0], refresh_memory=False)
        else:
            self._learn_batch(new_ids)

        self._refresh_memory()
        if self.autosave_path:
            self.save(self.autosave_path)
        return len(prepared)

    def read(self, record_id: str) -> Any:
        if record_id not in self.records:
            raise KeyError(record_id)
        return self.records[record_id]

    def update(self, record_id: str, record: Any):
        if record_id not in self.records:
            raise KeyError(record_id)
        self._check_json(record)

        self.records[record_id] = record
        self.features[record_id] = self._make_features(record)
        self._learn_online(
            record_id,
            train_sequence=False,
            advance_stream=False,
        )
        # Updates should not make an older record the newest item in the stream.
        self._rebuild_stream_state()
        if self.autosave_path:
            self.save(self.autosave_path)

    def delete(self, record_id: str):
        if record_id not in self.records:
            raise KeyError(record_id)

        del self.records[record_id]
        del self.features[record_id]
        self.memory_keys.pop(record_id, None)
        self.feature_keys.pop(record_id, None)
        self.record_importance.pop(record_id, None)
        self._refresh_memory()
        self._rebuild_stream_state()
        if self.autosave_path:
            self.save(self.autosave_path)

    @torch.no_grad()
    def _rebuild_stream_state(self):
        """Recompute sequential state from surviving records in insertion order."""
        self.model.eval()
        state = torch.zeros(self.latent_dim)
        last_state_before = None
        last_latent = None
        rebuilt_importance: dict[str, float] = {}

        for rid, (fields, mask) in self.features.items():
            z = self.model.encode(fields.unsqueeze(0), mask.unsqueeze(0)).squeeze(0)
            last_state_before = state.clone()
            state, importance = self.model.advance_state_with_importance(state, z)
            state = state.detach().clone()
            last_latent = z.detach().clone()
            rebuilt_importance[rid] = float(importance.item())

        self.stream_state = state
        self.last_state_before = last_state_before
        self.last_latent = last_latent
        self.record_importance = rebuilt_importance

    def filter(self, predicate) -> list[dict[str, Any]]:
        return [
            {"id": rid, "record": record}
            for rid, record in self.records.items()
            if predicate(rid, record)
        ]

    def all_records(self) -> list[dict[str, Any]]:
        return [
            {"id": rid, "record": record}
            for rid, record in self.records.items()
        ]

    # ---------------- Associative retrieval ----------------

    @torch.no_grad()
    def search(
        self,
        query: Any,
        top_k: int = 5,
        latent_weight: float = 0.8,
    ) -> list[dict[str, Any]]:
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        if not 0.0 <= latent_weight <= 1.0:
            raise ValueError("latent_weight must be in [0, 1]")
        if not self.records:
            return []

        q_fields, q_mask = self._make_features(query)
        q_z = self._embed(q_fields, q_mask)
        q_raw = (q_fields * q_mask.unsqueeze(-1)).sum(dim=0)
        q_raw = F.normalize(q_raw, p=2, dim=0)

        ids = self._memory_ids
        latent_scores = self._memory_matrix @ q_z
        feature_scores = self._feature_matrix @ q_raw
        scores = (
            latent_weight * latent_scores
            + (1.0 - latent_weight) * feature_scores
        )

        # Stable ordering preserves insertion order for equal-scoring records.
        ranked_indices = torch.argsort(scores, descending=True, stable=True)
        results = []
        for index in ranked_indices[:top_k].tolist():
            rid = ids[index]
            results.append({
                "id": rid,
                "record": self.records[rid],
                "score": float(scores[index].item()),
                "latent_score": float(latent_scores[index].item()),
                "feature_score": float(feature_scores[index].item()),
                "importance": self.record_importance.get(rid, 0.5),
            })
        return results

    @torch.no_grad()
    def predict_next_latent(self) -> list[float]:
        self.model.eval()
        z = self.model.predict_next(self.stream_state.unsqueeze(0))
        return z.squeeze(0).tolist()

    # ---------------- Persistence ----------------

    def save(self, path: str = "neuralstore.pt"):
        checkpoint = {
            "version": self.VERSION,
            "config": {
                "field_dim": self.field_dim,
                "max_fields": self.max_fields,
                "hidden_dim": self.hidden_dim,
                "latent_dim": self.latent_dim,
                "learning_rate": self.optimizer.param_groups[0]["lr"],
                "replay_batch_size": self.replay_batch_size,
                "online_steps": self.online_steps,
                "autosave_path": self.autosave_path,
            },
            "records_json": json.dumps(self.records, allow_nan=False),
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "memory_keys": self.memory_keys,
            "record_importance": self.record_importance,
            "stream_state": self.stream_state,
            "last_state_before": self.last_state_before,
            "last_latent": self.last_latent,
            "training_steps": self.training_steps,
            "last_loss": self.last_loss,
        }
        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path: str = "neuralstore.pt") -> "NeuralStore":
        checkpoint = torch.load(
            path, map_location="cpu", weights_only=True
        )

        if checkpoint.get("version") != cls.VERSION:
            raise ValueError("Unsupported checkpoint version")

        config = checkpoint["config"]
        store = cls(
            field_dim=config["field_dim"],
            max_fields=config["max_fields"],
            hidden_dim=config["hidden_dim"],
            latent_dim=config["latent_dim"],
            learning_rate=config["learning_rate"],
            replay_batch_size=config["replay_batch_size"],
            online_steps=config["online_steps"],
            autosave_path=config.get("autosave_path"),
        )

        store.records = json.loads(checkpoint["records_json"])
        store.features = {
            rid: store._make_features(record)
            for rid, record in store.records.items()
        }

        store.model.load_state_dict(checkpoint["model_state"])
        store.optimizer.load_state_dict(checkpoint["optimizer_state"])

        store.stream_state = checkpoint["stream_state"].float()
        last_state_before = checkpoint.get("last_state_before")
        store.last_state_before = (
            last_state_before.float() if last_state_before is not None else None
        )
        last_latent = checkpoint.get("last_latent")
        store.last_latent = last_latent.float() if last_latent is not None else None
        store.record_importance = checkpoint.get("record_importance", {})
        store.training_steps = checkpoint["training_steps"]
        store.last_loss = checkpoint["last_loss"]

        store.model.eval()
        store._refresh_memory()
        return store