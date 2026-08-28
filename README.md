# EEG-fMRI-BCG

BCGNet GRU (ECG→BCG) on FASTR-corrected EEG. AAS and PCA-OBS are separate comparators, never part of training.

| Command | Config | Output |
|---------|--------|--------|
| `bcgnet discover` | `config.yaml` | subject/recording list with run labels |
| `bcgnet run` | `config.yaml` | `*_fastr_bcgnet.vhdr`, `training_history.png`, before/after PSD |
| `bcgnet aas` | `compare.yaml` | `*_fastr_bcg.vhdr` |
| `bcgnet pca-obs` | `compare.yaml` | `*_fastr_pcaobs.vhdr` |
| `bcgnet compare` | `compare.yaml` | Raw/AAS/PCA-OBS/BCGNet PSD and epoch plots, `compare_summary.csv` |

Python 3.12: `uv sync`

## Configuration

`config.yaml` (training) and `compare.yaml` (comparison) at the repository root are the live configs for this study, tracked so the paths that produced a result stay on record. `examples/` holds the same two files with placeholder paths; copy those when setting up elsewhere. A relative `bcgnet_config` resolves against the folder holding the compare config, so the root `compare.yaml` points at the root `config.yaml`.

### Recording labels

Every recording is labelled from its own filename, never from its position in
the folder. A name carrying a run token (`run1`, `run-02`, `RUN.3`) is that run;
a name carrying none — a baseline or resting acquisition — is not a run at all.
It is still trained on and still cleaned, it is just named after its own
filename (`BaselineEEG`) and left out of the run count, so a subject missing
run 1 is reported as missing it rather than having its baseline promoted into
the gap. Staged files, figures, and the summary columns all use that label:
`sub0000_run2_raw.vhdr`, `psd_BaselineEEG_avg.png`.

Studies that number runs some other way set their own regex, whose first group
is the run number:

```yaml
naming:
  run_pattern: '_S(\d+)_'
```

## BCGNet

```text
bcgnet discover --config config.yaml
bcgnet run --config config.yaml
```

Input is FASTR EEG (gradient gone, BCG still present). One model per subject, trained at 100 Hz. The BCG estimate is interpolated and subtracted from the original 1 kHz EEG (`bcgnet.writeback`), so ECG and line noise are unchanged. Channel names match the FASTR file (e.g. `FPz`).

BCGNet uses no cardiac markers: the vendor dataset cuts fixed 3 s epochs and learns ECG→EEG directly, so R-peak detection never enters this path.

`save_figures` writes `training_history.png` and a Before/After PSD (Raw vs BCGNet only). Comparator overlays are not produced here.

Vendor GRU details and compatibility patches: [`docs/UPSTREAM.md`](docs/UPSTREAM.md).

## Comparators

Independent R-peak detection and both bounded corrections live in `src/bcg_correction/`. The two arms share one detector, one window, and one delay; they differ only in how the artifact estimate is built.

```text
bcgnet aas     --config compare.yaml
bcgnet pca-obs --config compare.yaml
```

Each arm writes to its own root with its own filename suffix (`_bcg` for AAS, `_pcaobs` for PCA-OBS), so the two never overwrite each other. Method notes: [`docs/aas/bcg_methods.md`](docs/aas/bcg_methods.md).

The single-recording entry point is unchanged:

```text
bcg-correct correct-bcg --config examples/aas/bcg_correction.yml
```

## Compare

With every `run.*` flag false, this only plots folders you already have:

```text
bcgnet compare --config compare.yaml
```

- FASTR: `paths.fastr_root`
- AAS: `paths.aas_root` (`*_fastr_bcg.vhdr`)
- PCA-OBS: `paths.pca_obs_root` (`*_fastr_pcaobs.vhdr`)
- BCGNet: `paths.bcgnet_root` (`*_fastr_bcgnet.vhdr`)

Arms missing from disk are simply left out of the overlay; `compare_summary.csv` keeps the same columns either way, blank where an arm did not run. Heartbeat-locked residuals are scored against the R train recorded by whichever bounded arm ran — both write the same detector output to `<stem>.bcg.json`.

`prefer_comparator` is true when BCGNet remaining power in delta, theta, or alpha is `> 1`, or heartbeat-locked residual is worse than raw. Alpha-peak collapse is reported but does not set that flag.

## Citation

McIntosh, J. R., Yao, J., Hong, L., Faller, J., & Sajda, P. (2020). Ballistocardiogram artifact reduction in simultaneous EEG-fMRI using deep learning. IEEE Transactions on Biomedical Engineering.
