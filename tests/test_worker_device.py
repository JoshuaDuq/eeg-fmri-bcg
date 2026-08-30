"""The environment a training worker runs in, derived from ``compute``.

``process_subject`` sets these before importing TensorFlow, so they are
computed as plain data here rather than exercised through a real training run.
"""

from pathlib import Path

import pytest
import yaml

from bcgnet.cohort import require_device, worker_environment
from bcgnet.config import ConfigurationError, load_config


def _config(tmp_path: Path, device: str | None = None):
    compute = {"workers": 2, "cpu_count": 10, "threads_per_worker": 5}
    if device is not None:
        compute["device"] = device
    document = {
        "paths": {
            "fastr_root": str(tmp_path / "fastr"),
            "output_root": str(tmp_path / "out"),
        },
        "compute": compute,
        "training": {
            "num_epochs": 80,
            "es_patience": 12,
            "batch_size": 1,
            "learning_rate": 0.001,
            "random_seed": 1997,
            "architecture": "default_rnn_model",
            "overwrite": True,
            "resume": True,
            "save_model": True,
            "save_data": True,
            "save_figures": True,
        },
        "preprocess": {
            "new_fs": 100,
            "len_epoch": 3,
            "mad_threshold": 5,
            "per_training": 0.7,
            "per_valid": 0.15,
            "per_test": 0.15,
            "ecg_channel": "ECG",
        },
        "subjects": {"include": [], "exclude": []},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return load_config(path)


def test_cpu_worker_hides_every_gpu(tmp_path: Path) -> None:
    environment = worker_environment(_config(tmp_path, "cpu"))
    assert environment["CUDA_VISIBLE_DEVICES"] == "-1"


def test_gpu_worker_exposes_the_configured_card(tmp_path: Path) -> None:
    environment = worker_environment(_config(tmp_path, "gpu:1"))
    assert environment["CUDA_VISIBLE_DEVICES"] == "1"


def test_gpu_worker_enables_memory_growth(tmp_path: Path) -> None:
    """Without this each worker claims the whole card and the second one OOMs.

    The vendor tree asks for growth through a TF1 ``ConfigProto``, which Keras 3
    ignores, so the environment variable is what actually takes effect.
    """
    environment = worker_environment(_config(tmp_path, "gpu"))
    assert environment["TF_FORCE_GPU_ALLOW_GROWTH"] == "true"


def test_cpu_worker_does_not_ask_for_memory_growth(tmp_path: Path) -> None:
    environment = worker_environment(_config(tmp_path, "cpu"))
    assert "TF_FORCE_GPU_ALLOW_GROWTH" not in environment


def test_worker_threads_come_from_the_config(tmp_path: Path) -> None:
    environment = worker_environment(_config(tmp_path))
    assert environment["OMP_NUM_THREADS"] == "5"
    assert environment["TF_NUM_INTRA_OP_THREADS"] == "5"
    assert environment["TF_NUM_INTER_OP_THREADS"] == "1"


def test_require_device_rejects_a_gpu_run_with_no_card_visible(
    tmp_path: Path,
) -> None:
    """The silent trap: TensorFlow on native Windows is CPU-only from 2.11 on.

    Without this the cohort trains for hours on CPU having been asked for a GPU.
    """
    config = _config(tmp_path, "gpu")
    with pytest.raises(ConfigurationError, match="no GPU"):
        require_device(config.compute, visible_gpus=[])


def test_require_device_accepts_a_gpu_run_with_a_card_visible(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path, "gpu")
    require_device(config.compute, visible_gpus=["/physical_device:GPU:0"])


def test_require_device_ignores_missing_cards_on_a_cpu_run(tmp_path: Path) -> None:
    config = _config(tmp_path, "cpu")
    require_device(config.compute, visible_gpus=[])
