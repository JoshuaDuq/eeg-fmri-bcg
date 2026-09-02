import json
from pathlib import Path

import pytest
import yaml
from test_cardiac_markers import make_source_recording

import bcg_correction.bcg_benchmark as benchmark_module
from bcg_correction.bcg_benchmark import BenchmarkSummary
from bcg_correction.brainvision import read_brainvision_markers
from bcg_correction.cardiac_markers import (
    PULSE_MARKER_DESCRIPTION,
    PULSE_MARKER_TYPE,
)
from bcgstudy.cli import main


def test_detect_cardiac_command_executes_independent_marker_pipeline(
    tmp_path: Path,
    capsys,
) -> None:
    source_vhdr, _ = make_source_recording(tmp_path)
    config_path = tmp_path / "cardiac.yml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "input": {"vhdr": str(source_vhdr)},
                "output": {"vhdr": str(tmp_path / "detected.vhdr")},
                "detector": {
                    "ecg_channel": "ECG",
                    "preprocessing_band_hz": [7.0, 40.0],
                    "teager_emphasis_hz": 10.0,
                    "teager_smoothing_seconds": 0.028,
                    "template_window_seconds": [-0.2, 0.4],
                    "minimum_rr_seconds": 0.4,
                    "maximum_rr_seconds": 1.5,
                    "candidate_refractory_seconds": 0.25,
                    "candidate_prominence_mad": 3.0,
                    "correlation_threshold": 0.5,
                    "refinement_iterations": 2,
                },
            }
        ),
        encoding="utf-8",
    )

    assert main(["detect", "--config", str(config_path)]) == 0

    summary = json.loads(capsys.readouterr().out)
    assert summary["marker_count"] == 5
    _, markers = read_brainvision_markers(tmp_path / "detected.vmrk")
    assert sum(
        marker.marker_type == PULSE_MARKER_TYPE
        and marker.description == PULSE_MARKER_DESCRIPTION
        for marker in markers
    ) == 5


def test_benchmark_bcg_command_loads_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    config_path = tmp_path / "benchmark.yml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "benchmark": {
                    "fastr_root": str(tmp_path / "fastr"),
                    "analyzer_input_root": str(tmp_path / "analyzer_input"),
                    "analyzer_output_root": str(tmp_path / "analyzer_output"),
                    "output_root": str(tmp_path / "reports"),
                    "marker_tolerance_seconds": 0.05,
                    "correction_methods": ["aas", "pca_obs"],
                    "correction_window_seconds": [-0.1, 0.3],
                    "ecg_to_bcg_delay_seconds": 0.21,
                    "aas_neighbor_count": 20,
                    "pca_obs_components": 3,
                    "cross_fit_fold_count": 2,
                    "null_surrogate_count": 10,
                    "random_seed": 42,
                },
                "detector": {
                    "ecg_channel": "ECG",
                    "preprocessing_band_hz": [7.0, 40.0],
                    "teager_emphasis_hz": 10.0,
                    "teager_smoothing_seconds": 0.028,
                    "template_window_seconds": [-0.2, 0.4],
                    "minimum_rr_seconds": 0.4,
                    "maximum_rr_seconds": 1.5,
                    "candidate_refractory_seconds": 0.25,
                    "candidate_prominence_mad": 3.0,
                    "correlation_threshold": 0.5,
                    "refinement_iterations": 2,
                },
            }
        ),
        encoding="utf-8",
    )
    expected = BenchmarkSummary(
        report_json=tmp_path / "reports" / "bcg_benchmark.json",
        report_csv=tmp_path / "reports" / "bcg_benchmark.csv",
        run_count=0,
        successful_count=0,
        failed_count=0,
    )
    captured = {}

    def fake_run(config):
        captured["fastr_root"] = config.fastr_root
        captured["methods"] = config.correction_methods
        return expected

    # Patch where it is defined: the CLI imports lazily, so there is no
    # module-level name on the command module to replace.
    monkeypatch.setattr(benchmark_module, "run_bcg_benchmark", fake_run)

    assert main(["benchmark", "--config", str(config_path)]) == 0

    assert captured["fastr_root"] == (tmp_path / "fastr").resolve()
    assert captured["methods"] == ("aas", "pca_obs")
    assert json.loads(capsys.readouterr().out)["run_count"] == 0


def test_help_text_is_study_independent(capsys) -> None:
    with pytest.raises(SystemExit):
        main(["--help"])

    help_text = capsys.readouterr().out.lower()
    assert "participant" not in help_text
    assert "thermal" not in help_text
