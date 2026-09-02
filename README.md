# EEG-fMRI-BCG

Four ballistocardiogram correction methods for FASTR-corrected EEG — **AAS**,
**PCA-OBS**, **blocked mean**, and **BCGNet** — plus a comparison that measures
them against each other on the same recordings.

No method is the default. Run whichever one your study needs, or run all four
and let `bcg compare` show you which behaves best on your data.

## Install

```bash
uv sync
```

Python 3.12.

## Quickstart

Each pipeline is independent — none requires the others to have run.

```bash
# 1. See what will be processed, without processing it
uv run bcg discover --config config.yaml

# 2. Pick a correction — any one, in any order, none depends on the others
uv run bcg aas     --config compare.yaml    # average artifact subtraction
uv run bcg pca-obs --config compare.yaml    # optimal basis set
uv run bcg blocked-mean --config compare.yaml  # cross-fitted mean template
uv run bcg bcgnet  --config config.yaml     # BCGNet GRU, trains per subject

# 3. Compare whatever you ran
uv run bcg compare --config compare.yaml
```

Changed a correction or comparison figure and want it applied to work you
already did? Rebuild those reports from outputs on disk, without redoing a
single correction or retraining BCGNet:

```bash
uv run bcg reports --config compare.yaml
```

Correct a single recording without a cohort:

```bash
uv run bcg correct --config examples/bcg_correction.yml
```

## What each command writes

| command | config | output |
|---------|--------|--------|
| `bcg discover` | `config.yaml` | subject/recording list with run labels, to stdout |
| `bcg aas` | `compare.yaml` | `*_fastr_aas.vhdr` + reports, under `paths.aas_root` |
| `bcg pca-obs` | `compare.yaml` | `*_fastr_pcaobs.vhdr` + reports, under `paths.pca_obs_root` |
| `bcg blocked-mean` | `compare.yaml` | `*_fastr_blockedmean.vhdr` + reports, under `paths.blocked_mean_root` |
| `bcg bcgnet` | `config.yaml` | `*_fastr_bcgnet.vhdr`, one model per subject, `cohort_summary.csv` |
| `bcg compare` | `compare.yaml` | `compare_summary.csv` + method-vs-method pages in `paths.experiments_root` |
| `bcg reports` | `compare.yaml` | bounded-arm and comparative reports rebuilt from existing outputs |

Arms write to separate roots with distinct filename suffixes, so they never
overwrite each other and any subset can exist at once.

## Reports

Every figure this project produces uses the same six panels, so learning to read
one teaches you to read all of them. Only the averaging changes.

| level | where | answers |
|-------|-------|---------|
| recording | `<stem>_correction_report.png`, beside the corrected file | what did this correction do here? |
| subject | `<arm_root>/reports/<bids_id>_<suffix>_report.png` | is this subject usable, are the runs comparable? |
| cohort | `<arm_root>/reports/cohort_<suffix>_report.png` | how did this method do overall? |
| methods | `<experiments_root>/cohort_comparative.png` and `subjects/` | which method behaves best on this data? |
| topography | `cohort_<suffix>_topography.png` per arm, `<experiments_root>/cohort_topography.png` across arms | **where on the scalp** each method acts |

| panel | shows |
|-------|-------|
| A | heartbeat-locked artifact, before vs after |
| B | what was removed, against the artifact |
| C | spectrum before vs after |
| D | **what was removed, split cardiac-locked vs not** |
| E | residual left behind (aggregate) or a time excerpt (single recording) |
| F | the numbers |

Subject and cohort pages are grand averages of the small `*_profile.npz` each
correction leaves beside its output, so they cost no re-reading of EEG.

### How to read them

**Panel D decides everything.** Cardiac harmonics are phase-locked by
definition, so anything a method removed that is *not* phase-locked cannot be
BCG. Energy there inside the shaded alpha band is neural signal on its way out.

This matters because **a method that subtracts more than the artifact posts the
best residual ratio precisely because it is also removing brain**. Suppression
alone cannot tell a good correction from an aggressive filter. Read the residual
ratio, the specificity, and the alpha collateral together:

| number | meaning | good |
|--------|---------|------|
| `locked residual ratio` | heartbeat-locked energy left, after ÷ before | low |
| `specificity` | share of the removal that was cardiac-locked | high (1.0 = surgical) |
| `alpha collateral` | share of the recording's alpha removed in the non-locked part | low |

The comparative page adds **spectral specificity** — the locked share of the
removal at each frequency, bounded 0–1. It is the clearest single view of
over-subtraction: a method dipping inside the alpha band is taking signal there.

The topography page makes the same case spatially, and needs no advance guess
about which signal is at risk. BCG is peripheral and motion-driven; the alpha
rhythm is posterior. Row A shows the artifact and where each method removed it;
row B shows the alpha rhythm and where each method took collateral. **A
collateral map shaped like the alpha map is signal loss** — and the test
generalises to any spatially structured signal a correction might eat, because
it asks whether the removal is organised like brain activity at all.

Every other figure collapses 63 channels to one trace, so this is the only place
that information survives.

`locked residual ratio` on a page is the same number as `residual_qc.ratio` in
the provenance JSON and `locked_*_ratio` in `compare_summary.csv`; a test asserts
the page cannot disagree with the provenance beside it.

## Configuration

`config.yaml` (training) and `compare.yaml` (correction and comparison) at the
repository root are the live configs for this study, tracked so the paths that
produced a result stay on record. `examples/` holds the same files with
placeholder paths — copy those when setting up elsewhere. Relative paths inside
a config resolve against the folder holding it.

### Recording labels

Every recording is labelled from its own filename, never from its position in
the folder. A name carrying a run token (`run1`, `run-02`, `RUN.3`) is that run;
a name carrying none — a baseline or resting acquisition — is not a run at all.
It is still processed, it is just named after its own filename (`BaselineEEG`)
and left out of the run count, so a subject missing run 1 is reported as missing
it rather than having its baseline promoted into the gap.

Studies that number runs some other way set their own regex, whose first group
is the run number:

```yaml
naming:
  run_pattern: '_S(\d+)_'
```

### Parallelism

`compute.workers` sets how many recordings or subjects run at once. The binding
constraint is RAM, not cores: a bounded-arm worker holds one 1 kHz recording, a
BCGNet worker holds a whole subject. Raise it against memory.

### CPU or GPU (BCGNet only)

Training is CPU-only unless `compute.device` says otherwise:

```yaml
compute:
  workers: 1
  device: gpu      # or "gpu:1" to pick a card on a multi-GPU host
```

A `gpu` run fails immediately when TensorFlow reports no card, rather than
spending hours on the CPU having been asked for the GPU. That failure is the
common case on Windows: TensorFlow has shipped no native-Windows GPU build since
2.10, so `pip install tensorflow` there is CPU-only and silent about it. Run
under WSL2 with `tensorflow[and-cuda]`.

Every GPU worker also gets `TF_FORCE_GPU_ALLOW_GROWTH`. The vendor tree asks for
memory growth through a TF1 `ConfigProto` that Keras 3 ignores, so without it the
first worker claims the whole card and the rest fail to start.

## Methods

Independent R-peak detection and the bounded corrections live in
`src/bcg_correction/`. The bounded arms share one detector, one window, and
one delay; they differ only in how the artifact estimate is built. Method notes:
[`docs/bcg_methods.md`](docs/bcg_methods.md).

**BCGNet** (`src/bcgnet/`) trains one GRU per subject at 100 Hz, then
interpolates the BCG estimate and subtracts it from the original 1 kHz EEG, so
ECG and line noise are unchanged. It uses no cardiac markers — the vendor dataset
cuts fixed 3 s epochs and learns ECG→EEG directly, so R-peak detection never
enters that path. Vendor details and compatibility patches:
[`docs/UPSTREAM.md`](docs/UPSTREAM.md).

### RR gaps

An RR interval longer than `detector.maximum_rr_seconds` means a beat was not
detected. No template is subtracted where there is no beat, so that span keeps
full-amplitude BCG — the correction elsewhere is unaffected. Such spans are
written to the output as `Bad_BCG` "Bad Interval" markers for downstream epoch
rejection, and their share is reported as `rr_gap_fraction` in the arm's
`<stem>.bcg.json`.

`correction.maximum_gap_fraction` caps that share. Above it the recording is
refused: a recording whose gaps dominate is not one with a few uncorrected spans,
it is one whose *detected* beats cannot be trusted either.

Every other detection degradation stays fatal, and `rr_below_minimum` is why the
two differ: an interval shorter than physiology means a **spurious** detection,
which subtracts a template where no beat occurred and so injects artifact. A gap
leaves data uncorrected; a spurious beat makes data wrong, and only the first has
a bounded downstream remedy.

`correction.residual_floor_uv` is the companion escape hatch for
`maximum_residual_ratio`. The ratio is scale-free, so a recording that began with
little heartbeat-locked energy cannot halve it however well the correction ran;
below the floor the ratio stops gating output.

## A caveat on `cohort_summary.csv`

`bcg bcgnet` reports the vendor's own numbers: `rms_raw`/`rms_bcgnet` and the
per-band `*_bcgnet_ratio` columns. These are **total power** ratios, not
artifact-reduction measurements. A band ratio of 0.45 says that band's power
halved; it does not say whether BCG left or neural signal did.

Do not read a low `alpha_bcgnet_ratio` as signal loss: BCG is harmonically
related to heart rate and its harmonics land squarely in 8–13 Hz, so alpha
attenuation is an expected consequence of correction. `compare.qc` computes
`alpha_peak_collapsed` for information and deliberately keeps it out of
`prefer_comparator` for that reason.

Artifact suppression is measured heartbeat-locked, and only the correction and
compare paths do it. `bcg_correction.metrics` holds the stricter forms:
`held_out_cardiac_rms` scores even beats against the odd-beat template and back,
so no beat helps build the template that scores it, and
`circular_shifted_cardiac_null` supplies the surrogate null. Signal transfer is a
separate measurement on injected known signals (`tone_transfer`,
`band_rms_ratio`, `event_locked_rms_ratio`).

## Development

```bash
uv run pytest
uv run ruff check src tests
uv run bcg discover --config config.yaml
uv run python tools/bench_bcgnet.py     # per-sample GRU timing
```

## Layout

```
src/bcgstudy/         neutral CLI, discovery, and bounded-arm orchestration
src/bcg_correction/   R detection, AAS, PCA-OBS, blocked mean, metrics, report figures
src/bcgnet/           BCGNet training and write-back
src/bcgnet/compare/   the only place arms are read against each other
docs/                 method notes and vendor patches
examples/             configs with placeholder paths
experiments/          method-vs-method figures for this study
tools/                standalone utilities
```

Every pipeline is a sibling subcommand of `bcg`. There is deliberately no
default method and no command named after one — running AAS through a command
called `bcgnet`, as an earlier layout did, quietly tells users which method is
the real one.

## Citation

McIntosh, J. R., Yao, J., Hong, L., Faller, J., & Sajda, P. (2020).
Ballistocardiogram artifact reduction in simultaneous EEG-fMRI using deep
learning. *IEEE Transactions on Biomedical Engineering*.
