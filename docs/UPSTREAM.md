# Upstream BCGNet

Wrapper around [jiaangyao/BCGNet](https://github.com/jiaangyao/BCGNet) (McIntosh et al., IEEE TBME 2020). The GRU lives in `src/bcgnet/vendor/` with the original script layout (`from config import get_config`, `from session import Session`).

Paper: https://ieeexplore.ieee.org/document/9124646

Upstream trains a `Session` on `{subject}_r0{run}_raw.set`. This package stages FASTR BrainVision into that naming, then `load_all_dataset` → `prepare_training` → `train` → `clean` → `evaluate`.

## Vendor patches

Upstream targets Python 3.8 / TensorFlow 2.3 / MNE 0.20. Patches are compatibility only:

- SciPy `median_absolute_deviation` → `median_abs_deviation`
- Keras 3 `learning_rate` (not `lr`); no `save_format=` on `save_weights`
- GRU `implementation=2` omitted when Keras rejects it
- float32 instead of float64
- BrainVision `.vhdr` and FIF in addition to EEGLAB `.set`
- rename `FPz`/`CPz` to names in `default_config.yaml` **during training**; export restores FASTR names
- `reject_by_annotation=False` so MNE 1.x does not drop FASTR volume markers
- Keras `Sequence.__init__` and `predict` without unsupported `callbacks`
- empty path strings in `default_config.yaml` are not forced through `Path()`
- raw matplotlib strings for µV labels (Python 3.12)

Do not install `vendor/requirement.yml`. Use this package's `pyproject.toml`.

## Study overrides

Paper YAML is the reference (`num_epochs: 2500`, `batch_size: 1`, `es_patience: 25`, `new_fs: 100`, 3 s epochs). `config.yaml` carries those values unchanged, batch size included. `es_min_delta` (1e-5) is not exposed by the wrapper and stays at the vendor default.

Upstream interpolates the whole 100 Hz residual back to the original rate. This wrapper interpolates **only the BCG estimate** and subtracts it from the original 1 kHz EEG (`bcgnet.writeback`), then writes BrainVision (`*_fastr_bcgnet.vhdr`).

`bcg bcgnet` writes corrected BrainVision, one model per subject, and `cohort_summary.csv`. It draws no correction figures: BCGNet uses no cardiac markers, so the heartbeat-locked panels the report system is built on need an R train, which only `bcg compare` has. BCGNet therefore appears in the method-vs-method pages, scored against the detector output written by whichever bounded arm ran.
