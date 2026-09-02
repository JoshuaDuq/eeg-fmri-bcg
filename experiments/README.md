# experiments

Method-vs-method figures for this study, written by `bcgnet compare` into the
path named by `paths.experiments_root` in `compare.yaml`.

| file | what it is |
|------|------------|
| `cohort_comparative.png` | every arm that ran, grand-averaged over the cohort |
| `cohort_topography.png` | where on the scalp each arm took artifact, and where it took collateral |
| `subjects/<bids_id>_comparative.png` | the same six panels for one participant |

Six panels, one trace per arm, no arm privileged — the order and colours come
from `src/bcgnet/compare/arms.py`, so a new method appears here as soon as it is
declared there.

| panel | shows |
|-------|-------|
| A | the heartbeat-locked artifact, uncorrected and after each arm |
| B | the residual each arm leaves, on its own scale, with the interquartile band across recordings |
| C | power spectrum after each arm |
| D | removed power that is **not** cardiac-locked, i.e. collateral |
| E | at each frequency, the share of the removed power that was cardiac-locked |
| F | every recording as a point: locked ratio (ECG regressed), locked ratio as written, alpha collateral |

One legend per page; an arm keeps its colour in every panel and every figure.
Panel titles are descriptive; the reading guide is the caption at the foot.
Each cohort page is written as PNG and as PDF (vector, editable text).

## How to read them

Panels D and E are the pair that matters. Cardiac harmonics are phase-locked by
definition, so anything an arm removed that is *not* phase-locked cannot be BCG.
An arm sitting high in panel D inside the shaded alpha band is removing neural
signal — and it will be doing that while posting the **best** residual ratio in
panel A, because subtracting more than the artifact lowers that ratio too.

Two cautions the metrics themselves cannot express:

- A beat-invariant template scores near-zero collateral and near-perfect
  specificity *by construction*, whatever its shape. Those two numbers measure
  how adaptive a method is; they do not show a non-adaptive method is right.
- Every metric except the ratio marked *as written* regresses the ECG channel
  out first. That is what keeps volume-conducted QRS from counting as BCG, but
  it hides ECG-shaped residual a method leaves in the file. A gap between the
  two ratios in panel F is that residual.

So never rank on the residual ratio alone. Read the three columns of panel F
together, then choose the arm that suits your analysis: a study that depends on
alpha-band power has different priorities from one that only needs the artifact
gone.

Per-recording and per-arm pages live beside the corrected data in each arm's own
output root, not here.

## Topography

Every other figure collapses 63 channels into one trace. `cohort_topography.png`
is the only place the spatial information survives, and it carries the argument
that needs no assumption about which band matters:

- **Row A** — the artifact (peripheral, motion-driven) and where each arm removed
  it, on one shared amplitude scale.
- **Row B** — the alpha rhythm (posterior, own scale) and the share of it each arm
  removed as collateral at every channel, on one scale across arms.

An arm whose row-B map resembles the alpha reference beside it is removing brain.

## Rebuilding

These figures come from the `*_profile.npz` beside each corrected recording, so
`bcg reports --config compare.yaml` regenerates all of them in minutes without
recomputing a single correction.
