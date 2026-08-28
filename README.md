# EEG-fMRI-BCG

One project, two BCG methods, plus a comparison command.

| Command | What it does | Figures |
|---------|--------------|---------|
| `bcgnet run` | Train one GRU per subject, write `*_fastr_bcgnet.vhdr` | `training_history.png` and before/after PSD |
| `bcgnet aas` | AAS on every FASTR recording | none |
| `bcgnet compare` | Overlay FASTR vs AAS vs BCGNet | PSD + epoch + `compare_summary.csv` |

They do not share correction code. Training never loads AAS. Before/after PSDs
are Raw vs BCGNet only; AAS overlays stay in `bcgnet compare`.

## Install

Python 3.12.

```text
uv sync
```

## BCGNet (GRU)

```text
bcgnet discover --config config.yaml
bcgnet run --config config.yaml
```

Input is FASTR-corrected EEG (gradient gone, BCG still present). One model per
subject. Details: [`docs/UPSTREAM.md`](docs/UPSTREAM.md).

## AAS (bundled BCG-Python)

The former BCG-Python library lives in `src/bcg_correction/`. Same YAML as
before, plus a cohort runner:

```text
bcg-correct correct-bcg --config examples/aas/bcg_correction.yml
bcgnet aas --config examples/compare.yaml
```

Method notes: [`docs/aas/bcg_methods.md`](docs/aas/bcg_methods.md).

## Compare the two

With both `run.aas` and `run.bcgnet` false, this only plots folders you already
have (`bcgnet run` / `bcgnet aas`):

```text
bcgnet compare --config examples/compare.yaml
```

That is the only Raw / AAS / BCGNet overlay.

It reads:

- FASTR: `paths.fastr_root`
- AAS: `paths.aas_root` (`*_fastr_bcg.vhdr`)
- BCGNet: `paths.bcgnet_root` (`*_fastr_bcgnet.vhdr`)

and writes PSD/epoch overlays plus `compare_summary.csv` (delta/theta/alpha
remaining, heartbeat-locked residual, `prefer_aas` when BCGNet adds power or
worsens locked residual).

To generate a method first, set `run.aas` and/or `run.bcgnet`.

## Citation

McIntosh, J. R., Yao, J., Hong, L., Faller, J., & Sajda, P. (2020).
Ballistocardiogram artifact reduction in simultaneous EEG-fMRI using deep
learning. IEEE Transactions on Biomedical Engineering.
