# Upstream BCGNet

This package wraps [jiaangyao/BCGNet](https://github.com/jiaangyao/BCGNet)
(McIntosh, Yao, Hong, Faller, Sajda, IEEE TBME 2020). The GRU that maps ECG →
BCG lives in `src/bcgnet/vendor/` and keeps the original script-package layout
(`from config import get_config`, `from session import Session`).

Paper: https://ieeexplore.ieee.org/document/9124646

## Original layout (GitHub)

```
BCGNet
|-config      default_config.yaml, load_config.py
|-dataset     Dataset object for one run
|-models      paper GRU (default_models.py) and custom-arch hook
|-session     Session training loop and data generator
|-utils       random-seed and stdout helpers
|-demo.ipynb  original Jupyter tutorial (not shipped)
|-example_data EEGLAB .set demo (not shipped)
```

The original tutorial trains one `Session` on EEGLAB files named
`{subject}_r0{run}_raw.set`. This package stages FASTR BrainVision recordings
into that same naming convention, then calls the same Session API:
`load_all_dataset` → `prepare_training` → `train` → `clean` → `evaluate`.

## Why the vendor copy is patched

The GitHub tree targets Python 3.8, TensorFlow 2.3, CUDA 10.1, MNE 0.20, and
SciPy 1.4. That stack does not run on this machine. Patches in `vendor/` are
limited to compatibility:

- SciPy `median_absolute_deviation` → `median_abs_deviation`
- Keras 3 `learning_rate` (not `lr`), no `save_format=` on `save_weights`
- GRU `implementation=2` omitted when the installed Keras rejects it
- float32 instead of float64 (Apple Silicon / modern TensorFlow)
- load BrainVision `.vhdr` and FIF in addition to EEGLAB `.set`
- rename `FPz`/`CPz` to the names in `default_config.yaml`
- `reject_by_annotation=False` so MNE 1.x does not drop FASTR volume markers
- Keras `Sequence.__init__` and `predict` without unsupported `callbacks`
- empty path strings in `default_config.yaml` are not forced through `Path()`
- raw matplotlib strings for µV axis labels (Python 3.12 `SyntaxWarning`)

Paper hyperparameters in the upstream YAML remain the reference
(`num_epochs: 2500`, `batch_size: 1`, `es_patience: 25`, `new_fs: 100`,
3 s epochs). The study `config.yaml` may use shorter training for a cohort
run; those values override the vendor YAML at Session construction.

Do not install `vendor/requirement.yml`. Use this package's `pyproject.toml`.
