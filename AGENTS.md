# Agent instructions

## No method is the default

AAS, PCA-OBS, blocked mean, and BCGNet are four correction methods on equal
footing. Nothing
in the code, config, or docs should present one as the product and the others as
also-rans — a user picks the arm their study needs, and `bcg compare` is how
they decide. Neutral naming included: `tests/bcg_correction/` covers the bounded
arms, not just AAS.

The one asymmetry that is real and must stay: **never feed a corrected output
back into BCGNet training.** Training input is FASTR EEG.

## One entry point

`bcg` (`src/bcgstudy/cli.py`) is the only command. Every pipeline is a sibling
subcommand: `discover aas pca-obs blocked-mean bcgnet compare correct detect
benchmark`.
Never add a command named after a method, and never give one a default.

## Keep the stacks separate

- `src/bcgstudy/` — the `bcg` command. Orchestration only; deliberately not
  named after any method.
- `src/bcg_correction/` — independent R detection, the bounded corrections
  (AAS, PCA-OBS, blocked mean), metrics, and the report figures. Reached via
  `bcg aas`, `bcg pca-obs`, `bcg blocked-mean`, `bcg correct`.
- `src/bcgnet/` — BCGNet training and write-back. Loads no comparator.
- `src/bcgnet/compare/` — the only place arms are read against each other. May
  call `run`/`aas`/`pca-obs`/`blocked-mean`; do not mix internals.

Comparison arms are declared once in `src/bcgnet/compare/arms.py`. Adding or
changing an arm means editing that module and the compare config — never a
hardcoded method tuple, label, or filename suffix elsewhere. An arm also needs a
colour in `figure_style.ARM_COLORS`; without one it is drawn in the fallback and
`tests/test_compare_arms.py` fails. Every `Arm.suffix` is the
arm's own method name; outputs written before an arm was renamed have to be
renamed on disk to stay discoverable.

`compare_summary.csv` is one rectangular table: every arm contributes the same
columns whether or not it ran.

## Figures

One visual language, defined in `src/bcg_correction/figure_style.py` and imported
by every figure module, so pages cannot drift apart. Okabe-Ito palette — the
comparison must stay readable for colour-vision deficiency, which rules out
green-versus-red.

Three report levels share the same six panels; only the averaging changes.
Recording and subject/cohort pages come from `correction_report.py`, method-vs-
method pages from `compare/comparative.py`. Subject and cohort pages average the
`*_profile.npz` each correction writes, so they never re-read EEG.

Topography is not optional decoration: it is the only figure where the spatial
structure survives, and it is what distinguishes removing an artifact from
removing a signal without having to guess the band in advance. It is rendered by
the same shared function for a single method run and for a comparison.

`bcg reports` rebuilds every figure from corrections already on disk. Keep that
true — changing a figure must never require redoing a correction.

Do not add a figure that shows total band power as a quality measure. Total power
cannot separate artifact removal from signal loss, and reading it that way is the
specific mistake this project's metrics exist to prevent. Suppression is always
reported beside specificity.

## Everything else

FASTR-Python is gradient-only (`mri-correct`).

Prefer the vendor Session API in `src/bcgnet/vendor/` over rewriting the network.
Record compatibility patches in `docs/UPSTREAM.md`.

Study knobs belong in YAML, not environment variables or hardcoded paths.
`config.yaml` and `compare.yaml` at the root are the live study configs;
`examples/` holds the placeholder templates. Keep both in step when the schema
changes — `tests/test_shipped_configs.py` enforces it. When a threshold is chosen
from data, record the evidence in the config comment beside it.

```text
uv run pytest
uv run ruff check src tests
uv run bcg discover --config config.yaml
```
