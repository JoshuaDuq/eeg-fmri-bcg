# Agent instructions

Keep the two correction stacks separate:

- `src/bcgnet/` — product. `bcgnet run` trains the GRU, write-back subtracts BCG from 1 kHz FASTR EEG, writes BrainVision. No AAS load. Figures: `training_history.png` and Raw vs BCGNet PSD.
- `src/bcg_correction/` — independent R detection and AAS (`bcgnet aas` / `bcg-correct`). PCA-OBS exists in this library; do not wire it into study compare.
- `src/bcgnet/compare/` — the only Raw vs AAS vs BCGNet overlay. May call `run`/`aas`; do not mix internals.

FASTR-Python is gradient-only (`mri-correct`).

Prefer the vendor Session API in `src/bcgnet/vendor/` over rewriting the network. Record compatibility patches in `docs/UPSTREAM.md`.

Study knobs belong in YAML (`config.yaml`, `examples/compare.yaml`), not environment variables or hardcoded paths.

```text
uv run pytest
uv run ruff check src tests
uv run bcgnet discover --config config.yaml
```
