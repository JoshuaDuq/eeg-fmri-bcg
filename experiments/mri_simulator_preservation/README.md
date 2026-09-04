# Paired thermal 8-13 Hz concordance (MRI simulator vs scanner)

Same thermal protocol in an MRI simulator (no B0, no gradients) and in the
scanner after FASTR. The simulator is an outside-field **concordance
reference** for the thermal 8-13 Hz change, not neural ground truth.

This follows Allen et al. (1998): compare physiological spectral responses
outside vs inside, rather than ranking methods on raw in-scanner power.
[Allen et al.](https://pubmed.ncbi.nlm.nih.gov/9758737/)
Later spectral-preservation work makes the same recommendation.
[Spectral-preservation comparison](https://pubmed.ncbi.nlm.nih.gov/36620439/)

## Estimand

The primary preservation endpoint is participant-level concordance of the
baseline-normalized posterior 8-13 Hz thermal response with the outside-field
simulator response. It is not neural ground truth and does not quantify all
alpha preservation. The week-separated reference includes test-retest,
vigilance, cap-placement, and impedance variability.

FASTR is the BCG-contaminated scanner reference. For each corrected method,
improvement_vs_fastr = E_fastr - E_method; positive values indicate greater
agreement with the simulator than the uncorrected scanner input. Cardiac-locked
residual must decrease at the same time. Neither endpoint is interpreted alone.

Trial inclusion is fixed using simulator reference quality and window
completeness before corrected methods are evaluated. Corrected output amplitude
never determines whether a trial enters the analysis.

Keep the label `8-13 Hz`; do not relabel it as alpha because scanner BCG
harmonics occupy the same band.

## What is compared

For each paired trial (participant, run, trial):

- prestimulus and plateau windows, posterior channels, 8-13 Hz band, and
  Welch segment length come from `config.yaml` (`ThermalProtocol`)
- Welch PSD via `Epochs.compute_psd()`, band integral, median across posterior
  channels

\[
R = 10\log_{10}(P_{\mathrm{plateau}} / P_{\mathrm{prestim}})
\]

Per participant and scanner series (FASTR, AAS, PCA-OBS, BCGNet):

\[
E = \mathrm{median}_{trials}|R_{\mathrm{series}} - R_{\mathrm{simulator}}|
\]

Trials are not treated as independent participants. Cardiac-locked residual
(existing 5-block ECG-regressed ratio on the same scanner runs) is tabulated
beside \(E\). FASTR's residual is 1 by definition. A method has to be closer
to the simulator than FASTR **and** reduce cardiac-locked artifact.

## Cohort and inclusion ledger

Configured subjects and runs live in `config.yaml`. Every participant/run is
written to `run_inclusion.csv` with one of:

- `no_completed_simulator`
- `missing_trial_summary`
- `sequence_mismatch`
- `missing_fastr`
- `missing_corrected_arm`
- `included`

`sub-0008` is excluded in the YAML (temperature/surface sequence differed).
A run is included only when simulator and scanner TrialSummary sequences match
trial-for-trial, the EEG has 11 `Trig_therm` markers, and FASTR plus AAS,
PCA-OBS, and BCGNet all exist on the same time grid.

The primary trial mask is simulator window completeness, simulator posterior
peak-to-peak (200 µV, a priori), and FASTR window completeness. BCG itself
can exceed 200 µV, so that threshold is not applied to FASTR or to corrected
outputs.

## Outputs

Numbers, not strip plots of five participants:

| file | contents |
|------|----------|
| `results/trial_pairs.csv` | every kept trial, FASTR and all three methods |
| `results/participant_summary.csv` | \(E\), signed difference, improvement vs FASTR, cardiac residual |
| `results/method_summary.csv` | participant-median \(E\), improvement, residual |
| `results/run_inclusion.csv` | every configured run decision |

Figures are spectra:

- `thermal_response_spectra` — \(R(f)\) for the simulator, FASTR, and each method
- `prestim_plateau_psd` — Allen-style prestimulus vs plateau PSD (descriptive;
  absolute level is not a ranking)

## How to run

```bash
uv run python experiments/mri_simulator_preservation/run.py \
  --config experiments/mri_simulator_preservation/config.yaml
```

`--limit N` measures the first N paired runs (smoke). The inclusion ledger
still records every configured run.
