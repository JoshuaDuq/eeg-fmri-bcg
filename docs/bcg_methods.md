# Independent cardiac detection and BCG correction

Library for `src/bcg_correction/` (`bcg aas`, `bcg pca-obs`, `bcg correct`).
`bcg bcgnet` does not call it. Both bounded methods below are study compare
arms: AAS writes `*_fastr_aas.vhdr` and PCA-OBS writes `*_fastr_pcaobs.vhdr`,
each under its own root, so no arm overwrites another.

This stage is independent of BrainVision Analyzer's cardiac markers. It
accepts FASTR gradient-corrected BrainVision and detects cardiac events from the ECG
samples alone.

## Input boundary

The FASTR input must not contain an exact `Pulse Artifact,R` marker. Existing unrelated
markers—such as `New Segment`, volume, synchronization, and stimulus markers—are
preserved. The detector does not read BrainVision annotations, and it does not use
Analyzer markers as seeds, gap boundaries, or correction events.

The output is a new BrainVision copy. It preserves the source marker collection and
appends one zero-based detector sample as one BrainVision `Pulse Artifact,R` marker.
No source pulse markers are removed or replaced. If a pre-existing exact pulse marker is
found in the FASTR input, the run fails before writing output; this prevents accidental
use of Analyzer-derived input as an allegedly independent FASTR input.

The correction output also includes `<stem>_psd_before.png` and
`<stem>_psd_after.png`. Both figures use the same longest contiguous corrected
window and use MNE's 0--100 Hz raw PSD renderer. Historical
`Bad Interval/Bad_Gradient` annotations are removed from the PSD copy because
the FASTR stage has corrected those intervals; every other MNE ``Bad``
annotation remains active. These spectra are supplementary; the acceptance
gate uses heartbeat-locked RMS rather than broadband PSD.

## Detector

The detector is a deterministic single-channel ECG procedure with four explicit stages:

1. robust centering and scale normalization;
2. a locked, zero-phase cardiac morphology band (0.5--10 Hz in the supplied study
   configuration; the band is part of the provenance and is not inferred from Analyzer);
3. nonnegative k-Teager energy with short smoothing and an adaptive
   moving-energy/slope/rhythm gate; and
4. QRS morphology correlation, refractory consolidation, interval validation, local
   alignment, and missing-event recovery. Recovery fills a gap only when the
   interval exceeds 1.7 times the recording's median RR. It searches 0.25 of
   that typical RR around each expected time, scores candidates against a
   template from nearby accepted beats, and keeps recoveries through the final
   event selection. Slightly long physiological RR (for example 1.58 s) is not
   filled at half-period; T-waves and empty pauses are left unmarked so
   remaining excess RR can be annotated ``Bad_BCG``.

This follows the signal-processing family used by the FMRIB fMRI QRS detector and the
MRI-specific validation of Niazy et al. It is a reimplementation of the method family,
not a copy of GPL MATLAB source. The `pain_study` recovery code informed template,
double-mark, physiological-interval, and held-out scoring decisions, but its
Analyzer-seeded gap-recovery path is not called by this production detector.

The installed FMRIB/EEGLAB implementation was run separately as a method audit. On the
available FASTR diagnostic, its unchanged QRS detector produced an implausibly dense event
train, and its installed OBS wrapper changed the ECG channel. It is therefore documented
as a reference implementation rather than a production dependency or an independent
source of ground-truth markers.

The returned markers are ECG R-sample positions. They are not assumed to be the peak of
the BCG artifact. Correction estimates one recording-level ECG-to-BCG delay from the
FASTR EEG and those R samples. The delay is scored on posterior EEG channels
after removing the contemporaneous linear ECG projection, so volume-conducted
QRS is not mistaken for BCG. The search is a 0--400 ms grid in 10 ms steps,
using median heartbeat-locked RMS in a +/-50 ms window around the delayed
anchor. The YAML
`ecg_to_bcg_delay_seconds` value is recorded as the configured reference and is not
used as the applied delay. R markers stay at the ECG peak.

Detector quality remains an audit of every adjacent RR interval. `status` is `ok`
when every interval lies in `[minimum_rr_seconds, maximum_rr_seconds]` and
`degraded` when any interval does not, with the stable reasons
`rr_below_minimum`, `rr_above_maximum`, and `low_prominence_candidate`. The
study configuration uses `maximum_rr_seconds: 2.0` (30 bpm) so a slightly slow
but regular train is not treated as a gap.

BCG correction refuses fatal detector problems, including short RR intervals and
low-prominence candidates. Overlong RR gaps alone are tolerated only within
`maximum_gap_fraction`; they are annotated as bad intervals. Detector checks
precede writing. Residual metrics are descriptive and no longer gate acceptance.
The correction provenance records evaluation settings and measurements; the
same profile supplies its figures.

## Correction arms

The benchmark evaluates the bounded methods on the same FASTR input and detector
train:

- AAS forms a leave-one-out local template in Python. Every retained beat uses
  the complete fixed correction window. Neighbours are chosen from a temporal
  pool by epoch correlation, insufficient complete neighbour support is an error, and
  a per-channel least-squares scalar is fitted on constant-detrended epochs.
  A short cosine taper makes the subtraction zero at both window boundaries.
- PCA-OBS delegates the basis construction and per-event fit to MNE-Python's documented
  `apply_pca_obs` implementation.

Both methods modify EEG channels only inside the configured union of complete artifact
windows. The ECG channel and all samples outside that union are preserved exactly.

## Paired comparison

The benchmark requires three recordings for each pair: the FASTR gradient-corrected
input, Analyzer's gradient-corrected pre-BCG input, and Analyzer's BCG-corrected output.
The Analyzer input/output pair is necessary because the corrected file cannot be used as
the incumbent's own pre-correction baseline. The benchmark first detects events on FASTR
ECG, then validates channel order, sampling rate, and ECG identity against Analyzer's
pre-BCG input. Only after that check does it read Analyzer's exact `Pulse Artifact,R`
train for the incumbent correction arm and a one-to-one agreement audit. Analyzer markers
do not enter the independent detector or FASTR correction path.

Analyzer under-detection is expected in this cohort. Consequently, a lower independent
marker count is not required, and independent events absent from Analyzer are not
discarded merely because they disagree with its train. Analyzer agreement is reported as
an audit statistic, while the correction comparison uses each arm's actual pre-BCG input
and event train.

The report records the resolved inputs, SHA-256 hashes of all BrainVision sidecars,
detector quality, one-to-one marker agreement, correction anchors, method outputs, and
metrics. Existing reports and output recordings are never overwritten.

## Metrics and scientific limits

### Saved-output correction and comparison reports (profile schema 2)

For each recording and EEG channel, subtract its recording-level median.
Use the same original ECG peaks, delay-centred window, and complete epochs for
every arm. Primary measurements use the EEG as written; a separately labelled
sensitivity view regresses each trace against the same original ECG reference.
The difference between those views is not a uniquely identified cardiac field.

At resolution K, divide complete epochs chronologically into K near-equal
groups. For channel c, let m(c,b,t) be the mean within block b and n(b) its
beat count. Compute:

```text
local_RMS(c) = sqrt(sum_b n(b) * mean_t(m(c,b,t)^2) / sum_b n(b))
ratio = median_c(local_RMS_after(c)) / median_c(local_RMS_before(c))
```

Squaring before pooling prevents opposing block errors or channel polarities
from cancelling. The shipped K = 2/5/10/20 and minimum eight beats per block
come from the exploratory cancellation audit, not a validation of their
optimality. More blocks raise the finite-sample noise floor. No resolution is
an artifact-only measurement or a held-out test. Unsupported resolutions and
zero-denominator ratios are explicitly unavailable.

Pooled template curves are descriptive only. Waveform panels summarize
squared amplitudes across channels; scalar ratios use channel medians.
Removal phase-locking specificity is a descriptive energy fraction, not neural
specificity. Beat-variable removal can be variable artifact, neural signal, or
both; it is not even a guaranteed upper bound on neural loss, because
heartbeat-locked neural activity can also be removed. Spatial similarity to
alpha topography does not resolve that ambiguity.

The spectral locking fraction is computed within each recording by summing
locked and variable removal powers over the same EEG channels, then dividing.
Only those per-recording fractions enter participant aggregation; dividing
cohort-aggregated powers would give high-removal participants extra weight.

Welch spectra are calculated within individual epochs and averaged across
epochs, never over concatenated discontinuities. Density is integrated over
frequency for band-power units. Bands with insufficient frequency bins remain
undefined. Total band power in CSV exports is descriptive only and drives no
quality flag or method preference.

Subject and cohort comparison pages use the intersection of recording
identities across the compared arms. Cohort summaries first average within
participant, then take participant medians. IQR bars describe participant
variation, not inferential confidence intervals. Unsupported measurements
propagate as unavailable rather than changing the contributing sample.
Coverage includes missing outputs and unavailable profiles, not inferred
correction failures.

If there is no paired cohort, rebuilding fails explicitly rather than leaving
old figures looking current. An unpairable subject with an existing report also
raises a stale-report error; files are not silently deleted.

Every arm reports `preservation_status=not_measured`. A zeroed signal achieves
zero residual but does not thereby preserve brain activity. Preservation
comparisons require matched fitting/evaluation protocols: frozen BCGNet
injection transfer cannot be ranked against refitted bounded corrections.
Data already used for tuning remain exploratory, including after changing
the metrics. No automatic winner or residual rejection gate is warranted.

`bcg reports` recomputes profiles from the original FASTR and existing corrected
files without running any correction or network. Old profile schemas are
rejected. Single-method cohort pages consume the rebuilt profiles, not EEG.

Implementation references: [MNE PCA-OBS example](https://mne.tools/stable/auto_examples/preprocessing/esg_rm_heart_artefact_pcaobs.html),
[SciPy Welch estimator](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html),
and [scikit-learn grouped evaluation](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data).
The local-block diagnostic is this project's explicitly tested measurement,
not a claim of endorsement or ground-truth validation by those packages.

### Separate Analyzer benchmark

The following describes the existing Analyzer agreement benchmark, not the
saved-output comparison metrics above.

The primary residual measure is held-out cardiac RMS: even beats are scored against an
odd-beat template and odd beats against an even-beat template. The event being scored
therefore cannot contribute to its own template. A deterministic circular-shift null
retains the event interval structure while destroying its phase alignment to the EEG.

The report also includes per-channel residual ratios, the null maximum, ECG preservation,
and the maximum EEG change outside the correction windows. Lower residual alone is not
evidence of superiority; signal transfer and manual review remain necessary.

Analyzer agreement is not ground truth. A defensible sensitivity or positive-predictive
value claim requires blinded manual adjudication of a stratified ECG subset. Without
that adjudication, the benchmark supports only the explicitly reported agreement,
held-out residual, null, and preservation measurements.

## References

- [Niazy et al. (2005), FMRIB fMRI artifact reduction](https://doi.org/10.1016/j.neuroimage.2005.06.067)
- [FMRIB/EEGLAB QRS reference implementation](https://github.com/sccn/fMRIb)
- [MNE `find_ecg_events`](https://mne.tools/stable/generated/mne.preprocessing.find_ecg_events.html)
- [MNE `apply_pca_obs`](https://mne.tools/stable/generated/mne.preprocessing.apply_pca_obs.html)
- [Christov (2004), adaptive threshold ECG detection](https://pmc.ncbi.nlm.nih.gov/articles/PMC516783/)
- [Marino et al. (2018), adaptive OBS for BCG](https://www.nature.com/articles/s41598-018-27187-6)
- [Systematic review of EEG-fMRI artifact correction](https://pmc.ncbi.nlm.nih.gov/articles/PMC7991907/)
