"""The single neutral entry point over every correction method."""

from __future__ import annotations

import json

import pytest

from bcgstudy.cli import main


def test_help_lists_every_pipeline(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])
    assert exit_info.value.code == 0
    text = capsys.readouterr().out
    for command in ("discover", "aas", "pca-obs", "bcgnet", "compare",
                    "correct", "detect", "benchmark"):
        assert command in text


def test_no_method_is_the_default() -> None:
    """Every method is a sibling subcommand; none is implied by bare `bcg`."""
    with pytest.raises(SystemExit) as exit_info:
        main([])
    assert exit_info.value.code != 0


def test_missing_config_is_a_message_not_a_traceback(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["compare", "--config", "/does/not/exist.yaml"]) == 1
    assert "does not exist" in capsys.readouterr().err


@pytest.mark.parametrize(
    "command", ["discover", "aas", "pca-obs", "bcgnet", "compare",
                "correct", "detect", "benchmark"],
)
def test_every_command_requires_a_config(command: str) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main([command])
    assert exit_info.value.code != 0


def test_discover_prints_recording_labels(
    tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:
    import yaml

    fastr = tmp_path / "fastr" / "sub-0001"
    fastr.mkdir(parents=True)
    header = "Brain Vision Data Exchange Header File Version 1.0\n" + ("x" * 120)
    for name in ("BaselineEEG_sub0001_fastr", "Task_run1_sub0001_fastr"):
        (fastr / f"{name}.vhdr").write_text(header, encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({
        "paths": {"fastr_root": str(tmp_path / "fastr"),
                  "output_root": str(tmp_path / "out")},
        "compute": {"workers": 1, "cpu_count": 1},
        "training": {"num_epochs": 1, "es_patience": 1, "batch_size": 1,
                     "learning_rate": 0.001, "random_seed": 1,
                     "architecture": "default_rnn_model", "overwrite": True,
                     "resume": False, "save_model": False, "save_data": False,
                     "save_figures": False},
        "preprocess": {"new_fs": 100, "len_epoch": 3, "mad_threshold": 5,
                       "per_training": 0.7, "per_valid": 0.15, "per_test": 0.15,
                       "ecg_channel": "ECG"},
        "subjects": {"include": [], "exclude": []},
    }), encoding="utf-8")

    assert main(["discover", "--config", str(config)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert [r["label"] for r in payload[0]["recordings"]] == ["BaselineEEG", "run1"]
