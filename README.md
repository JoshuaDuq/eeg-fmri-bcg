# EEG-fMRI Ballistocardiogram (BCG) Correction

A scientific computing framework for the standardized correction and comparative evaluation of ballistocardiogram (BCG) artifacts in simultaneous EEG-fMRI recordings following gradient artifact suppression.

## Overview

Cardiovascular pulsatile activity inside high-field magnetic resonance scanners induces large-amplitude ballistocardiogram (BCG) artifacts on scalp electroencephalography. This software provides an integrated pipeline to benchmark and execute three correction methodologies under standardized physiological constraints:

1. **Average Artifact Subtraction (AAS)**: Moving-window sliding template subtraction aligned to detected electrocardiographic R-peaks (Allen et al., 1998).
2. **Principal Component Analysis Optimal Basis Set (PCA-OBS)**: Subspace projection onto principal components derived from cardiac-aligned artifact templates (Niazy et al., 2005).
3. **BCGNet (Recurrent Neural Network)**: Deep gated recurrent unit (GRU) network trained directly on single-lead electrocardiography (ECG) to predict multi-channel BCG artifacts without requiring cardiac peak markers (McIntosh et al., 2020).

No single method is treated as a default. Study pipelines evaluate each arm under unified validation criteria using `bcg compare`.

---

## Comparative evaluation

Suppression alone is not correction quality. The comparison uses identical
recordings, ECG events, windows, and EEG channels for every available arm.

- **Local heartbeat-locked RMS:** chronological equal-beat blocks are averaged
  separately, squared, and weighted by their beat counts before taking RMS.
  Per-channel amplitudes are summarized by their median. Ratios are amplitude
  ratios, not fractions of artifact energy.
- **Several resolutions:** the shipped configuration uses 2, 5, 10, and 20
  blocks, each requiring at least eight beats. These are exploratory
  sensitivity scales, not optimized settings or independent validation folds.
  Insufficiently supported scales are unavailable; zero reference amplitudes
  produce undefined ratios, not perfect scores.
- **Saved EEG and ECG-regressed sensitivity:** all EEG channels are evaluated.
  ECG is excluded from EEG RMS and spectra. The original ECG is the same
  regression reference for before and after.
- **Removal diagnostics:** heartbeat-locked removal fraction and beat-variable
  alpha removal describe what changed. Neither identifies artifact purity or
  neural loss. Variable BCG and heartbeat-evoked neural activity invalidate
  those interpretations.
- **Paired participant summaries:** recordings are intersected across the arms
  being compared before averaging within each participant. Cohort figures show
  participant medians; residual error bars show participant IQR, not confidence
  intervals. Missing output coverage is shown separately.

The same six panels serve recording, subject, cohort, and comparison reports:
pooled template RMS (cancellation-prone diagnostic), local RMS, multiscale
residual ratios, beat-variable removal spectra, removal phase-locking
specificity, and coverage/limitations. Topographies show local residuals on a
shared scale per resolution and descriptive variable-alpha removal.

There is **no automatic preferred method**, no power-reduction quality verdict,
and no residual-based output rejection threshold. Detector and gap safeguards
remain. Neural preservation is explicitly `not_measured`; matched known-signal
validation or independent task endpoints are needed to establish it. Freezing
BCGNet while refitting the other methods is not a matched preservation test.

The previous pooled-metric rankings and figures are not evidence under this
revised evaluation. Rebuild using existing FASTR and corrected EEG:

```bash
uv run bcg reports --config compare.yaml
```

This updates profiles and plots without training or applying any correction.
Schema-v1 profiles are rejected with rebuild instructions, not silently reused.
See [measurement definitions and limitations](docs/bcg_methods.md#metrics-and-scientific-limits).

---

## Installation

The package requires Python 3.12 and `uv` for reproducible environment management.

```bash
git clone https://github.com/JoshuaDuq/eeg-fmri-bcg.git
cd eeg-fmri-bcg
uv sync
```

---

## Execution & Workflow

All pipeline operations are accessed through the unified `bcg` command-line entry point (`src/bcgstudy/cli.py`). Subcommands execute independently and do not depend on preceding steps.

```bash
# 1. Discover recordings and verify subject-run mappings without processing
uv run bcg discover --config config.yaml

# 2. Execute desired correction pipelines
uv run bcg aas          --config compare.yaml
uv run bcg pca-obs      --config compare.yaml
uv run bcg bcgnet       --config config.yaml

# 3. Generate comparative evaluation tables and figures
uv run bcg compare --config compare.yaml

# 4. Recompute profiles and figures from saved EEG, without re-running corrections
uv run bcg reports --config compare.yaml
```

To correct a single BrainVision acquisition outside a cohort:
```bash
uv run bcg correct --config examples/bcg_correction.yml
```

### Subcommand Reference

| Command | Configuration | Output Description |
| :--- | :--- | :--- |
| `bcg discover` | `config.yaml` | Identifies available BIDS recordings and displays subject/run tables. |
| `bcg aas` | `compare.yaml` | Writes `*_fastr_aas.vhdr` and per-recording profiles under `paths.aas_root`. |
| `bcg pca-obs` | `compare.yaml` | Writes `*_fastr_pcaobs.vhdr` and per-recording profiles under `paths.pca_obs_root`. |
| `bcg bcgnet` | `config.yaml` | Trains subject GRU models; writes `*_fastr_bcgnet.vhdr` and `cohort_summary.csv`. |
| `bcg compare` | `compare.yaml` | Generates `compare_summary.csv`, comparative figures, and topographies. |
| `bcg reports` | `compare.yaml` | Recomputes profiles from saved EEG, then builds subject/cohort figures from profiles. |
| `bcg detect` | `compare.yaml` | Executes independent QRS detection on ECG channels. |
| `bcg benchmark` | `compare.yaml` | Computes signal transfer on synthetic calibration injections. |

---

## Signal Processing & Methodology

The repository enforces clean separation of concerns:
- **`src/bcgstudy/`**: Top-level CLI orchestration and dataset discovery.
- **`src/bcg_correction/`**: QRS detection (`cardiac.py`), bounded corrections (`bcg.py`), analytical metrics (`metrics.py`), and visualization (`figure_style.py`, `correction_report.py`).
- **`src/bcgnet/`**: Vendor GRU neural network training, inference, and sinc interpolation writeback (`export.py`, `writeback.py`).
- **`src/bcgnet/compare/`**: Multi-arm evaluation engine and comparative figure generation (`comparative.py`, `pipeline.py`).
- **`tools/`**: Statistical verification and holdout evaluation scripts.

### Key Processing Standards

1. **Non-destructive File Operations**: Corrected BrainVision datasets (`.vhdr`, `.eeg`, `.vmrk`) are written to separate output directories with explicit method suffixes (`_aas`, `_pcaobs`, `_bcgnet`). Raw and FASTR gradient-corrected inputs are never overwritten.
2. **Frequency Domain Alignment**: Bounded methods operate at native acquisition sampling rates (e.g., 1000 Hz). BCGNet trains on downsampled 100 Hz data and maps predicted BCG traces back to 1000 Hz via polyphase sinc interpolation prior to subtraction, preserving non-cardiac high frequencies.
3. **RR Interval Validation**: Heartbeat intervals below physiological limits ($RR < \text{minimum}$) indicate false detections and abort processing to prevent artifact injection. Prolonged intervals ($RR > \text{maximum}$) indicate missed beats and are annotated with `Bad_BCG` markers without subtracting unanchored templates.

---

## Configuration

Study parameters are maintained in dedicated YAML configuration files:
- **`config.yaml`**: BCGNet training parameters, architecture dimensions, and dataset roots.
- **`compare.yaml`**: Bounded correction parameters (epoch windows, delays, PCA components, neighbor counts) and comparison roots.
- **`examples/`**: Template configuration files with standardized placeholder directories.

---

## Verification & Testing

The test suite validates numerical equivalence, edge cases, and architectural boundaries across all modules:

```bash
uv run pytest
uv run ruff check src tests tools
```

All unit and integration tests must pass cleanly.

---

## References

- Allen, P. J., Polizzi, G., Krakow, K., Fish, D. R., & Lemieux, L. (1998). Identification of EEG events in the MR scanner: the problem of pulse artifact and a method for its subtraction. *NeuroImage*, 8(3), 229–239.
- Niazy, R. K., Xie, J., Miller, P., & Smith, S. M. (2005). Spectral and spatial characteristics of the ballistocardiogram artifact in simultaneous EEG-fMRI. *NeuroImage*, 28(3), 720–737.
- McIntosh, J. R., Yao, J., Hong, L., Faller, J., & Sajda, P. (2020). Ballistocardiogram artifact reduction in simultaneous EEG-fMRI using deep learning. *IEEE Transactions on Biomedical Engineering*, 68(1), 78–89.
