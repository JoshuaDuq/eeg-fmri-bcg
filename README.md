# BCGNet-Python

One project, two BCG methods, plus a comparison command.

| Namespace | Method | CLI |
|-----------|--------|-----|
| `bcgnet` | GRU ECG→BCG ([jiaangyao/BCGNet](https://github.com/jiaangyao/BCGNet)) | `bcgnet run` |
| `bcg_correction` | Independent R-peaks + AAS / PCA-OBS | `bcgnet aas` or `bcg-correct` |

They do not share correction code. Comparison only happens when you ask for it.

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

## AAS / PCA-OBS (bundled BCG-Python)

The former BCG-Python library lives in `src/bcg_correction/`. Same YAML as
before, plus a cohort runner:

```text
bcg-correct correct-bcg --config examples/aas/bcg_correction.yml
bcgnet aas --config examples/compare.yaml
```

Method notes: [`docs/aas/bcg_methods.md`](docs/aas/bcg_methods.md).

## Compare the two

`examples/compare.yaml` can either **run** a method or **reuse folders you
already have**. With both `run.aas` and `run.bcgnet` false it only plots:

```text
bcgnet compare --config examples/compare.yaml
```

That reads:

- FASTR: `paths.fastr_root`
- AAS: `paths.aas_root` (`*_fastr_bcg.vhdr`, e.g. `fastr_python_bcg_gapfix`)
- BCGNet: `paths.bcgnet_root` (`*_r0{n}_bcgnet.mat`)

and writes GitHub-style PSD and epoch overlays (Raw / AAS / BCGNet) plus
`compare_summary.csv`.

To compute a method first, set `run.aas: true` and/or `run.bcgnet: true`
(BCGNet then uses `bcgnet_config:`).

## Citation

McIntosh, J. R., Yao, J., Hong, L., Faller, J., & Sajda, P. (2020).
Ballistocardiogram artifact reduction in simultaneous EEG-fMRI using deep
learning. IEEE Transactions on Biomedical Engineering.
