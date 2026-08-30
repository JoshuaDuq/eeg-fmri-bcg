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


def _write_compare_config(tmp_path: Path, *, workers: int | None = None) -> Path:
    document = {
        "paths": {
            "fastr_root": str(tmp_path / "fastr"),
            "aas_root": str(tmp_path / "aas"),
            "pca_obs_root": str(tmp_path / "pca_obs"),
            "bcgnet_root": str(tmp_path / "bcgnet"),
            "output_root": str(tmp_path / "out"),
        },
        "run": {"aas": False, "pca_obs": False, "bcgnet": False},
        "correction": {
            "window_seconds": [-0.2, 0.7],
            "ecg_to_bcg_delay_seconds": 0.21,
            "aas_neighbor_count": 20,
            "pca_obs_components": 4,
            "maximum_residual_ratio": 0.5,
            "overwrite": False,
            "detector": {
                "ecg_channel": "ECG",
                "preprocessing_band_hz": [0.5, 10.0],
                "teager_emphasis_hz": 10.0,
                "teager_smoothing_seconds": 0.028,
                "template_window_seconds": [-0.2, 0.4],
                "minimum_rr_seconds": 0.4,
                "maximum_rr_seconds": 2.0,
                "candidate_refractory_seconds": 0.25,
                "candidate_prominence_mad": 2.0,
                "correlation_threshold": 0.5,
                "refinement_iterations": 2,
            },
        },
        "plot": {
            "channel": "Cz",
            "epoch_start_seconds": 10,
            "epoch_seconds": 3,
            "psd_max_hz": 30,
        },
        "subjects": {"include": [], "exclude": []},
    }
    if workers is not None:
        document["compute"] = {"workers": workers}
    path = tmp_path / "compare.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path


def _capture_batch(monkeypatch) -> dict:
    captured: dict = {}

    def fake_run(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(
        "bcgnet.correction_batch.run_correction_batch", fake_run
    )
    return captured


def test_pca_obs_command_writes_into_the_pca_obs_root(
    tmp_path: Path, monkeypatch
) -> None:
    from bcgnet.compare.arms import PCA_OBS

    captured = _capture_batch(monkeypatch)
    path = _write_compare_config(tmp_path)
    assert main(["pca-obs", "--config", str(path)]) == 0
    assert captured["arm"] is PCA_OBS
    assert captured["output_root"] == (tmp_path / "pca_obs").resolve()


def test_aas_command_still_writes_into_the_aas_root(
    tmp_path: Path, monkeypatch
) -> None:
    from bcgnet.compare.arms import AAS

    captured = _capture_batch(monkeypatch)
    path = _write_compare_config(tmp_path)
    assert main(["aas", "--config", str(path)]) == 0
    assert captured["arm"] is AAS
    assert captured["output_root"] == (tmp_path / "aas").resolve()


def test_comparator_commands_correct_recordings_in_parallel(
    tmp_path: Path, monkeypatch
) -> None:
    """``compute.workers`` must reach the batch, or the arms stay single-core."""
    captured = _capture_batch(monkeypatch)
    path = _write_compare_config(tmp_path, workers=4)

    assert main(["aas", "--config", str(path)]) == 0

    assert captured["workers"] == 4


def test_comparator_commands_stay_serial_without_a_compute_block(
    tmp_path: Path, monkeypatch
) -> None:
    """An existing compare config must keep the batch in one process."""
    captured = _capture_batch(monkeypatch)
    path = _write_compare_config(tmp_path)

    assert main(["pca-obs", "--config", str(path)]) == 0

    assert captured["workers"] == 1
