# Agent instructions

Keep the correction stacks separate:

- `src/bcgnet/` — product. `bcgnet run` trains the GRU, write-back subtracts BCG from 1 kHz FASTR EEG, writes BrainVision. No comparator load. Figures: `training_history.png` and Raw vs BCGNet PSD.
- `src/bcg_correction/` — independent R detection and the bounded corrections, AAS and PCA-OBS (`bcgnet aas`, `bcgnet pca-obs`, `bcg-correct`).
- `src/bcgnet/compare/` — the only Raw vs comparators vs BCGNet overlay. May call `run`/`aas`/`pca-obs`; do not mix internals.

BCGNet is the product; AAS and PCA-OBS are comparators. Never feed a comparator output into training.

Comparison arms are declared once in `src/bcgnet/compare/arms.py`. Adding or changing an arm means editing that module and the compare config — never a hardcoded method tuple, label, or filename suffix elsewhere. `Arm.suffix` for AAS stays `bcg` because recordings generated before PCA-OBS became its own arm use that name.

`compare_summary.csv` is one rectangular table: every arm contributes the same columns whether or not it ran.

FASTR-Python is gradient-only (`mri-correct`).

Prefer the vendor Session API in `src/bcgnet/vendor/` over rewriting the network. Record compatibility patches in `docs/UPSTREAM.md`.

Study knobs belong in YAML, not environment variables or hardcoded paths. `config.yaml` and `compare.yaml` at the root are the live study configs; `examples/` holds the placeholder templates. Keep both in step when the schema changes — `tests/test_shipped_configs.py` enforces it.

```text
uv run pytest
uv run ruff check src tests
uv run bcgnet discover --config config.yaml
```
