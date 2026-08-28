# Agent instructions for BCGNet-Python

Two correction libraries live here and must stay separate:

- `src/bcgnet/` — GRU ECG→BCG (`bcgnet run`); no AAS load. Writes
  `training_history.png` and Raw vs BCGNet (before/after) PSDs.
- `src/bcg_correction/` — independent R detection + AAS (`bcgnet aas`)
- `src/bcgnet/compare/` — the only Raw vs AAS vs BCGNet overlay; may *call*
  `run`/`aas`, never mix their internals

FASTR-Python stays gradient-only (`mri-correct`).

Prefer the Session API in `src/bcgnet/vendor/` over rewriting the network.
Keep vendor patches to compatibility with modern TensorFlow/Keras/MNE; record
them in `docs/UPSTREAM.md`.

Study knobs belong in `config.yaml`, not environment variables or hardcoded
paths in Python.

```text
uv run pytest
uv run ruff check src tests
uv run bcgnet discover --config config.yaml
```
