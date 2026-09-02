from pathlib import Path

import pytest
import yaml

from bcgnet.cohort import discover_subjects
from bcgnet.config import ConfigurationError, load_config


def _write_config(tmp_path: Path, **overrides) -> Path:
    document = {
        "paths": {
            "fastr_root": str(tmp_path / "fastr"),
            "output_root": str(tmp_path / "out"),
        },
        "compute": {
            "workers": 2,
            "cpu_count": 10,
            "threads_per_worker": "auto",
        },
        "training": {
            "num_epochs": 80,
            "es_patience": 12,
            "batch_size": 16,
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
    for key, value in overrides.items():
        document[key] = value
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path


def test_load_config_sets_auto_threads(tmp_path: Path) -> None:
    config = load_config(_write_config(tmp_path))
    assert config.compute.workers == 2
    assert config.compute.threads_per_worker == 5
    assert config.paths.fastr_root == (tmp_path / "fastr").resolve()


def test_workers_cannot_exceed_cpu_count(tmp_path: Path) -> None:
    path = _write_config(tmp_path)
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    document["compute"]["workers"] = 32
    document["compute"]["cpu_count"] = 10
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="cannot exceed"):
        load_config(path)


def test_unknown_field_is_rejected(tmp_path: Path) -> None:
    path = _write_config(tmp_path)
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    document["unexpected"] = True
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="unknown field"):
        load_config(path)


def test_eval_root_is_not_a_training_path(tmp_path: Path) -> None:
    path = _write_config(tmp_path)
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    document["paths"]["eval_root"] = str(tmp_path / "eval")
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="unknown field"):
        load_config(path)


def test_discover_lists_subject_folders(tmp_path: Path) -> None:
    fastr = tmp_path / "fastr" / "sub-0000"
    fastr.mkdir(parents=True)
    (fastr / "BaselineEEG_sub0000_fastr.vhdr").write_text(
        "Brain Vision Data Exchange Header File Version 1.0\n" + ("x" * 120),
        encoding="utf-8",
    )
    config = load_config(_write_config(tmp_path))
    subjects = discover_subjects(config)
    assert [spec["bids_id"] for spec in subjects] == ["sub-0000"]
    # The only recording here is a baseline, so it is not run 1 -- or any run.
    (recording,) = subjects[0]["recordings"]
    assert recording["label"] == "BaselineEEG"
    assert recording["run"] is None
    assert "eval_vhdr" not in recording


def test_run_pattern_defaults_when_naming_is_absent(tmp_path: Path) -> None:
    """An existing config predating the section keeps working unchanged."""
    from bcgstudy.discovery import DEFAULT_RUN_PATTERN

    config = load_config(_write_config(tmp_path))
    assert config.naming.run_pattern == DEFAULT_RUN_PATTERN


def test_run_pattern_can_be_overridden(tmp_path: Path) -> None:
    config = load_config(
        _write_config(tmp_path, naming={"run_pattern": r"_S(\d+)_"})
    )
    assert config.naming.run_pattern == r"_S(\d+)_"


def test_invalid_run_pattern_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="regular expression"):
        load_config(_write_config(tmp_path, naming={"run_pattern": "run(["}))


def test_run_pattern_without_a_capture_group_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="capture the run number"):
        load_config(_write_config(tmp_path, naming={"run_pattern": r"run\d+"}))


def test_unknown_naming_field_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="unknown field"):
        load_config(_write_config(tmp_path, naming={"task_pattern": "x"}))


def _with_device(tmp_path: Path, device: str) -> Path:
    return _write_config(
        tmp_path,
        compute={
            "workers": 2,
            "cpu_count": 10,
            "threads_per_worker": "auto",
            "device": device,
        },
    )


def test_device_defaults_to_cpu_when_absent(tmp_path: Path) -> None:
    """A config predating the knob keeps training on CPU, as it always did."""
    config = load_config(_write_config(tmp_path))
    assert config.compute.device == "cpu"
    assert config.compute.use_gpu is False
    assert config.compute.cuda_visible_devices == "-1"


def test_device_gpu_selects_the_first_card(tmp_path: Path) -> None:
    config = load_config(_with_device(tmp_path, "gpu"))
    assert config.compute.use_gpu is True
    assert config.compute.cuda_visible_devices == "0"


def test_device_gpu_index_selects_that_card(tmp_path: Path) -> None:
    config = load_config(_with_device(tmp_path, "gpu:1"))
    assert config.compute.use_gpu is True
    assert config.compute.cuda_visible_devices == "1"


def test_unknown_device_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match=r"compute\.device"):
        load_config(_with_device(tmp_path, "cuda"))


def test_device_gpu_with_a_non_numeric_index_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match=r"compute\.device"):
        load_config(_with_device(tmp_path, "gpu:first"))
