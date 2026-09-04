from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import yaml

from bcg_correction.bcg_config import (
    BenchmarkConfig,
    ConfigurationError,
    CorrectionRunConfig,
    DetectionRunConfig,
    load_benchmark_config,
    load_correction_config,
    load_detection_config,
)

DETECTOR_VALUES = {
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
}


def _detection_document() -> dict[str, object]:
    return {
        "input": {"vhdr": "data/gradient_corrected.vhdr"},
        "output": {"vhdr": "output/with_markers.vhdr"},
        "detector": DETECTOR_VALUES.copy(),
    }


def _benchmark_document() -> dict[str, object]:
    return {
        "detector": DETECTOR_VALUES.copy(),
        "benchmark": {
            "fastr_root": "/data/fastr_only",
            "analyzer_input_root": "/data/step1_scanner_artifact_pulse_marked",
            "analyzer_output_root": "/data/step3_bcg_corrected",
            "output_root": "/data/bcg_benchmark",
            "marker_tolerance_seconds": 0.1,
            "correction_methods": ["aas", "pca_obs"],
            "correction_window_seconds": [-0.2, 0.7],
            "ecg_to_bcg_delay_seconds": 0.210,
            "aas_neighbor_count": 20,
            "pca_obs_components": 4,
            "null_surrogate_count": 20,
            "random_seed": 20260826,
        },
    }


def _write_yaml(path: Path, document: dict[str, object]) -> Path:
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path


def test_load_detection_config_resolves_paths_and_values(tmp_path: Path) -> None:
    config = load_detection_config(
        _write_yaml(tmp_path / "detection.yml", _detection_document())
    )

    assert isinstance(config, DetectionRunConfig)
    assert config.input_vhdr == (tmp_path / "data/gradient_corrected.vhdr").resolve()
    assert config.output_vhdr == (tmp_path / "output/with_markers.vhdr").resolve()
    assert config.detector.ecg_channel == "ECG"
    assert config.detector.preprocessing_band_hz == (7.0, 40.0)
    assert config.detector.template_window_seconds == (-0.2, 0.4)


def test_load_correction_config_reads_method_and_window(tmp_path: Path) -> None:
    document = _detection_document()
    document["correction"] = {
        "method": "pca_obs",
        "window_seconds": [-0.2, 0.7],
        "ecg_to_bcg_delay_seconds": 0.21,
        "aas_neighbor_count": 20,
        "pca_obs_components": 4,
        "evaluation": {"block_counts": [2, 5, 10, 20], "minimum_beats_per_block": 8},
        "maximum_gap_fraction": 0.05,
    }
    config = load_correction_config(_write_yaml(tmp_path / "correction.yml", document))

    assert isinstance(config, CorrectionRunConfig)
    assert config.method == "pca_obs"
    assert config.window_seconds == (-0.2, 0.7)
    assert config.pca_obs_components == 4
    assert config.evaluation.block_counts == (2, 5, 10, 20)


def test_load_correction_config_rejects_unknown_method(tmp_path: Path) -> None:
    document = _detection_document()
    document["correction"] = {
        "method": "ica",
        "window_seconds": [-0.2, 0.7],
        "ecg_to_bcg_delay_seconds": 0.21,
        "aas_neighbor_count": 20,
        "pca_obs_components": 4,
        "evaluation": {"block_counts": [2, 5, 10, 20], "minimum_beats_per_block": 8},
        "maximum_gap_fraction": 0.05,
    }
    with pytest.raises(ConfigurationError, match="method"):
        load_correction_config(_write_yaml(tmp_path / "bad.yml", document))


def test_load_benchmark_config_preserves_absolute_roots(tmp_path: Path) -> None:
    config = load_benchmark_config(
        _write_yaml(tmp_path / "benchmark.yml", _benchmark_document())
    )

    assert isinstance(config, BenchmarkConfig)
    assert config.fastr_root == Path("/data/fastr_only")
    assert config.analyzer_input_root == Path(
        "/data/step1_scanner_artifact_pulse_marked"
    )
    assert config.analyzer_output_root == Path("/data/step3_bcg_corrected")
    assert config.ecg_to_bcg_delay_seconds == pytest.approx(0.210)
    assert config.correction_methods == ("aas", "pca_obs")


def test_benchmark_config_is_immutable(tmp_path: Path) -> None:
    config = load_benchmark_config(
        _write_yaml(tmp_path / "benchmark.yml", _benchmark_document())
    )

    with pytest.raises(FrozenInstanceError):
        config.ecg_to_bcg_delay_seconds = 0.0


@pytest.mark.parametrize(
    ("document_factory", "unknown_section", "unknown_field"),
    [
        (_detection_document, "detector", "unexpected"),
        (_benchmark_document, "benchmark", "unexpected"),
        (_benchmark_document, "detector", "unexpected"),
    ],
)
def test_config_rejects_unknown_fields(
    tmp_path: Path,
    document_factory,
    unknown_section: str,
    unknown_field: str,
) -> None:
    document = document_factory()
    document[unknown_section][unknown_field] = True

    with pytest.raises(ConfigurationError, match="unexpected"):
        loader = load_detection_config if "input" in document else load_benchmark_config
        loader(_write_yaml(tmp_path / "invalid.yml", document))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("preprocessing_band_hz", [40.0, 7.0], "preprocessing_band_hz"),
        ("template_window_seconds", [0.4, -0.2], "template_window_seconds"),
        ("minimum_rr_seconds", 1.5, "minimum_rr_seconds"),
        ("candidate_refractory_seconds", True, "candidate_refractory_seconds"),
        ("refinement_iterations", 0, "refinement_iterations"),
    ],
)
def test_detection_config_rejects_invalid_values(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    document = _detection_document()
    document["detector"][field] = value

    with pytest.raises(ConfigurationError, match=message):
        load_detection_config(_write_yaml(tmp_path / "invalid.yml", document))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("correction_methods", ["unknown"]),
        ("correction_methods", []),
        ("correction_window_seconds", [0.7, -0.2]),
        ("ecg_to_bcg_delay_seconds", -0.001),
        ("marker_tolerance_seconds", 0.0),
        ("aas_neighbor_count", 3),
        ("pca_obs_components", 0),
        ("null_surrogate_count", -1),
        ("random_seed", True),
    ],
)
def test_benchmark_config_rejects_invalid_values(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    document = _benchmark_document()
    document["benchmark"][field] = value

    with pytest.raises(ConfigurationError, match=field):
        load_benchmark_config(_write_yaml(tmp_path / "invalid.yml", document))


def test_config_does_not_require_input_paths_to_exist(tmp_path: Path) -> None:
    detection_path = _write_yaml(tmp_path / "detection.yml", _detection_document())
    benchmark_path = _write_yaml(tmp_path / "benchmark.yml", _benchmark_document())

    load_detection_config(detection_path)
    load_benchmark_config(benchmark_path)
