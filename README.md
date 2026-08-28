# EEG-fMRI-BCG

BCGNet GRU (ECG→BCG) on FASTR-corrected EEG. AAS is a separate comparator, not part of training.

| Command | Config | Output |
|---------|--------|--------|
| `bcgnet discover` | `config.yaml` | subject/run list |
| `bcgnet run` | `config.yaml` | `*_fastr_bcgnet.vhdr`, `training_history.png`, before/after PSD |
| `bcgnet aas` | `examples/compare.yaml` | `*_fastr_bcg.vhdr` |
| `bcgnet compare` | `examples/compare.yaml` | Raw/AAS/BCGNet PSD and epoch plots, `compare_summary.csv` |

Python 3.12: `uv sync`

## BCGNet

```text
bcgnet discover --config config.yaml
bcgnet run --config config.yaml
```

Input is FASTR EEG (gradient gone, BCG still present). One model per subject, trained at 100 Hz. The BCG estimate is interpolated and subtracted from the original 1 kHz EEG (`bcgnet.writeback`), so ECG and line noise are unchanged. Channel names match the FASTR file (e.g. `FPz`).

`save_figures` writes `training_history.png` and a Before/After PSD (Raw vs BCGNet only). AAS overlays are not produced here.

Vendor GRU details and compatibility patches: [`docs/UPSTREAM.md`](docs/UPSTREAM.md).

## AAS

Independent R-peak detection and average artifact subtraction live in `src/bcg_correction/`. Outputs are `*_fastr_bcg.vhdr`.

```text
bcgnet aas --config examples/compare.yaml
bcg-correct correct-bcg --config examples/aas/bcg_correction.yml
```

Method notes: [`docs/aas/bcg_methods.md`](docs/aas/bcg_methods.md). That library also implements PCA-OBS; **study compare does not use it.**

## Compare

With `run.aas` and `run.bcgnet` false, plots folders you already have:

```text
bcgnet compare --config examples/compare.yaml
```

- FASTR: `paths.fastr_root`
- AAS: `paths.aas_root` (`*_fastr_bcg.vhdr`)
- BCGNet: `paths.bcgnet_root` (`*_fastr_bcgnet.vhdr`)

`prefer_aas` is true when BCGNet remaining power in delta, theta, or alpha is `> 1`, or heartbeat-locked residual is worse than raw. Alpha-peak collapse is reported but does not set that flag.

## Citation

McIntosh, J. R., Yao, J., Hong, L., Faller, J., & Sajda, P. (2020). Ballistocardiogram artifact reduction in simultaneous EEG-fMRI using deep learning. IEEE Transactions on Biomedical Engineering.
