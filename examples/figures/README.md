# Per-participant BCGNet figures

Output of `bcgnet run --config config.yaml` with `save_figures` enabled — one
folder per participant, 165 PNGs from 21 participants.

| File | What it shows |
|------|---------------|
| `training_history.png` | Train/validation loss across epochs for that participant's GRU |
| `psd_<label>_avg.png` | Before/After PSD for one recording, averaged over channels — Raw (FASTR) vs BCGNet only |

The `<label>` in a PSD filename is the recording label taken from the source
filename, not its position in the folder: `run1`–`run6` for run-numbered
acquisitions, `BaselineEEG` for a recording carrying no run token. See
[Recording labels](../../README.md#recording-labels).

AAS and PCA-OBS never appear here. Comparator overlays come from
`bcgnet compare` and are written to the compare output root instead.

## Coverage

Every participant has `training_history.png`. Recording-level gaps are gaps in
the acquired data, not failed runs:

| Participant | Missing |
|-------------|---------|
| `sub0000` | `run1` |
| `sub0003` | `run3` |
| `sub0007` | `BaselineEEG` |

`sub0002` has no folder — that participant is absent from the dataset. All
other participants have all seven recordings.
