from pathlib import Path

import pytest
import yaml

from bcgnet.cli import main
from bcgnet.config import ConfigurationError, load_config


def _write_config(tmp_path: Path, **preprocess_overrides) -> Path:
    preprocess = {
        "new_fs": 100,
        "len_epoch": 3,
        "mad_threshold": 5,
        "per_training": 0.7,
        "per_valid": 0.15,
        "per_test": 0.15,
        "ecg_channel": "ECG",
    }
    preprocess.update(preprocess_overrides)
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
        "preprocess": preprocess,
        "subjects": {"include": [], "exclude": []},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path


def test_documented_cli_form_discovers_subjects(tmp_path: Path, capsys) -> None:
    fastr = tmp_path / "fastr" / "sub-0000"
    fastr.mkdir(parents=True)
    (fastr / "BaselineEEG_sub0000_fastr.vhdr").write_text(
        "Brain Vision Data Exchange Header File Version 1.0\n" + ("x" * 120),
        encoding="utf-8",
    )
    config_path = _write_config(tmp_path)
    assert main(["discover", "--config", str(config_path)]) == 0
    payload = capsys.readouterr().out
    assert "sub-0000" in payload


def test_split_fractions_must_sum_to_one(tmp_path: Path) -> None:
    path = _write_config(tmp_path, per_training=0.5, per_valid=0.5, per_test=0.5)
    with pytest.raises(ConfigurationError, match="sum to 1"):
        load_config(path)
