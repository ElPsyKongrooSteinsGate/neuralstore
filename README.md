# NeuralStore — Adaptive Neural Associative Memory

NeuralStore is an experimental hybrid neural database implemented in Python and PyTorch. It stores JSON-compatible records exactly in memory, learns compact vector representations from their fields, and ranks records by similarity to a query. It also maintains a recurrent state over the sequence of inserted or updated records.

> **Status:** educational prototype. The neural components are experimental; this is not a production database or a substitute for a vector database with a persistent index, transactions, or concurrency controls.

## Features

- Exact in-memory record operations: insert, read, update, delete, list, and predicate-based filtering.
- Deterministic feature encoding for nested dictionaries, lists/tuples, strings, numbers, booleans, and null values.
- Learned record embeddings with gated field pooling and an autoencoder-style reconstruction objective.
- Online optimization with replay samples from stored records.
- A recurrent state update with a learned importance score and a next-latent prediction head.
- Similarity search combining learned latent similarity and hashed-feature similarity.
- Checkpoint save/load, with optional save-after-mutation behavior.

## Requirements

- Python 3.10 or newer (the implementation uses modern type-annotation syntax).
- PyTorch.

Install PyTorch using the instructions appropriate for your operating system and hardware at [pytorch.org](https://pytorch.org/get-started/locally/). The dependency is listed in `requirements.txt`; `neuralstore.py` is the implementation module.

## Quick start

```python
from neuralstore import NeuralStore

store = NeuralStore(seed=7)
store.insert("user-001", {"name": "Ada", "role": "engineer", "active": True})
store.insert("user-002", {"name": "Linus", "role": "maintainer", "active": True})

print(store.read("user-001"))

matches = store.search({"role": "engineer"}, top_k=3)
for match in matches:
    print(match["id"], match["score"], match["record"])

store.save("neuralstore.pt")
restored = NeuralStore.load("neuralstore.pt")
```

Insertion and update perform online training, so they are more computationally expensive than ordinary dictionary operations. The first record has no preceding record for sequence-prediction training.

## Command-line usage

Run `python main.py` from Command Prompt to open the interactive prompt. It supports `add`, `import-json`, `read`, `update`, `delete`, `search`, `list`, and `quit`. Records and search queries are entered as JSON when prompted. The CLI automatically loads the checkpoint `neuralstore.pt` if it exists and saves it after changes.

One-shot commands are also available. Put `--db <checkpoint>` before the command to select a different checkpoint:

```text
python main.py add user-001 "{\"name\": \"Ada\", \"role\": \"engineer\"}"
python main.py import-json records.json
python main.py read user-001
python main.py search "{\"role\": \"engineer\"}" --top-k 3
python main.py list
```

Use `python main.py --help` to see the command options.

The `import-json` command accepts three JSON formats. The recommended format for one record is an object with an `id` and a `record` property. The `record` can contain any valid JSON data, including nested objects and arrays:

```json
{
    "id": "user-001",
    "record": {
        "name": "Ada",
        "role": "engineer",
        "active": true,
        "details": {
            "city": "London"
        }
    }
}
```

For multiple records, use a list of objects with `id` and `record` properties:

```json
[
    {"id": "user-001", "record": {"name": "Ada", "role": "engineer"}},
    {"id": "user-002", "record": {"name": "Linus", "role": "maintainer"}}
]
```

Alternatively, provide a JSON object mapping record IDs directly to record values:

```json
{
    "user-001": {"name": "Ada", "role": "engineer"},
    "user-002": {"name": "Linus", "role": "maintainer"}
}
```

The importer validates the file and rejects IDs that already exist rather than replacing stored records. Each ID must be a non-empty string, and record values must be valid JSON data. Import a file with:

```text
python main.py import-json records.json
```

### Import and search the included sample

The sample file at `sample_data/sampledata.json` contains one record with ID `sampledata`. From the project directory in Windows Command Prompt, import it into a separate checkpoint and search for its store ID:

```text
python main.py --db sample_neuralstore.pt import-json sample_data/sampledata.json
python main.py --db sample_neuralstore.pt search "{\"record_metadata\": {\"store_id\": \"NYC-FLAGSHIP-04\"}}" --top-k 3
python main.py --db sample_neuralstore.pt read sampledata
```

The `--db` option keeps this example's data in `sample_neuralstore.pt` rather than mixing it with the default `neuralstore.pt` database. Run the import command only once for that checkpoint; importing again will report that `sampledata` already exists.

## Technical specification

### Architecture

#### 1. `MixedFeatureEncoder`

Transforms one JSON-like record into a pair of tensors:

- `fields`: shape `[max_fields, field_dim]`, `float32`.
- `mask`: shape `[max_fields]`, `float32`; `1` indicates a populated field slot.

Nested dictionaries are flattened into dotted field paths; list entries share a `[]` path. Field and value tokens are mapped into fixed-size vectors using BLAKE2b hashing with signed contributions. Strings contribute a normalized category token, word tokens, and adjacent-word bigrams. Finite numbers contribute a presence feature and a bounded magnitude feature. Each field vector is L2-normalized.

The flattened fields are sorted deterministically and truncated to `max_fields`. Consequently, later fields can be omitted when a record has more than the configured limit, and hash collisions are possible by design.

#### 2. `NeuralModel`

The model contains:

- A shared per-field projection and sigmoid gate.
- Masked mean pooling and a projection into a normalized latent vector of size `latent_dim`.
- A decoder that reconstructs the mean input feature vector.
- An importance head that outputs a value in `[0, 1]`.
- A gated recurrent state transition.
- A predictor for the next latent representation.

The importance target is derived from the current record's reconstruction error, scaled and clamped to `[0, 1]`. During state updates, a larger importance value moves the retention gate toward preserving the previous state.

#### 3. `NeuralStore`

The store maintains separate in-memory mappings for original records, encoded features, learned memory keys, and importance values. `insert()` and `update()` run a configurable number of optimization steps using reconstruction loss, an adjacent-record prediction loss when a previous latent exists, and importance-estimation loss. The optimizer is AdamW and gradients are clipped.

Search computes cosine similarities between the query and each record in both latent space and encoded-feature space. The final score is:

`score = latent_weight * latent_score + (1 - latent_weight) * feature_score`

Results are sorted in descending score order. Similarity quality depends on the data, model dimensions, and training history; it is not guaranteed to represent semantic relevance.

### Public API

| Method | Behavior |
| --- | --- |
| `NeuralStore(...)` | Creates an empty store and model. |
| `insert(record_id, record)` | Adds a new record and trains online. Raises `ValueError` for an invalid ID or non-JSON-serializable value, and `KeyError` if the ID already exists. |
| `read(record_id)` | Returns the stored record; raises `KeyError` if absent. |
| `update(record_id, record)` | Replaces an existing record and trains online; raises `KeyError` if absent. |
| `delete(record_id)` | Removes an existing record; raises `KeyError` if absent. |
| `filter(predicate)` | Returns records for which `predicate(record_id, record)` is true. |
| `all_records()` | Returns all records as `{"id": ..., "record": ...}` dictionaries in insertion order. |
| `search(query, top_k=5, latent_weight=0.8)` | Returns up to `top_k` ranked matches, including component scores and importance. `latent_weight` must be in `[0, 1]`. |
| `predict_next_latent()` | Returns a list representing the predicted next latent vector from the current stream state. |
| `save(path="neuralstore.pt")` | Saves a checkpoint containing records, model/optimizer state, and training metadata. |
| `NeuralStore.load(path="neuralstore.pt")` | Restores a checkpoint created by the same supported code version. |

### Constructor defaults

`NeuralStore` defaults to `field_dim=64`, `max_fields=32`, `hidden_dim=96`, `latent_dim=32`, `learning_rate=1e-3`, `replay_batch_size=16`, `online_steps=3`, and `seed=7`. `autosave_path` defaults to `None`; when set, successful inserts, updates, and deletes save a checkpoint to that path.

### Persistence and ordering

Records are serialized as JSON in the checkpoint, while model and optimizer tensors are saved with PyTorch. Checkpoint loading currently requires the stored version to match the implementation's `VERSION`. Feature encoding is deterministic, but learned embeddings depend on model weights and training history. Stream state follows the store's record insertion order; deleting a record does not currently rebuild the recurrent state.

## Limitations and considerations

- All records and derived vectors are held in memory; search scans every record, so it is linear in the number of records.
- This implementation is single-process and provides no transaction, locking, or concurrent-write guarantees.
- `read()` returns the stored Python object directly. Mutating that object externally bypasses feature regeneration and model training; use `update()` to make changes visible to the store's derived representations.
- `delete()` removes the record and its associated feature/key entries, but does not retrain the model or rebuild the sequential state.
- Training occurs on insert/update and can be slow for large records or high `online_steps` values.
- Feature hashing can collide, and the fixed `max_fields` limit truncates records with many flattened fields.
- Importance is a learned reconstruction-surprise estimate, not a calibrated or human-verified measure of importance.
- Checkpoints should only be loaded from trusted sources. Keep backups of important data.

## License

No license is specified in this repository. Add a license file before distributing or reusing the project if a particular license is intended.
