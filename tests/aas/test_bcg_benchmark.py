from pathlib import Path

import numpy as np
import pytest

from bcg_correction.bcg_benchmark import (
    BenchmarkInputError,
    _RawSnapshot,
    _validate_pair_geometry,
    discover_recording_pairs,
)


def touch_recording(root: Path, name: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / name).write_text("header", encoding="utf-8")


def test_discover_recording_pairs_matches_runs_and_baselines(tmp_path: Path) -> None:
    fastr_root = tmp_path / "fastr"
    analyzer_input_root = tmp_path / "analyzer_input"
    analyzer_output_root = tmp_path / "analyzer_output"
    touch_recording(fastr_root, "ThermalPain_run2_sub0007_source.vhdr")
    touch_recording(fastr_root, "BaselineEEG_sub0007_source.vhdr")
    touch_recording(
        analyzer_input_root,
        "ThermalPain_run2_sub0007_input.vhdr",
    )
    touch_recording(
        analyzer_input_root,
        "BaselineEEG_sub0007_input.vhdr",
    )
    touch_recording(
        analyzer_output_root,
        "ThermalPain_run2_sub0007_output.vhdr",
    )
    touch_recording(
        analyzer_output_root,
        "BaselineEEG_sub0007_output.vhdr",
    )

    pairs = discover_recording_pairs(
        fastr_root,
        analyzer_input_root,
        analyzer_output_root,
    )

    assert [pair.recording_id for pair in pairs] == [
        "baseline_sub0007",
        "run2_sub0007",
    ]
    assert pairs[1].fastr_vhdr.name.endswith("source.vhdr")
    assert pairs[1].analyzer_input_vhdr.name.endswith("input.vhdr")
    assert pairs[1].analyzer_output_vhdr.name.endswith("output.vhdr")


def test_discover_recording_pairs_rejects_duplicate_keys(tmp_path: Path) -> None:
    fastr_root = tmp_path / "fastr"
    analyzer_input_root = tmp_path / "analyzer_input"
    analyzer_output_root = tmp_path / "analyzer_output"
    touch_recording(fastr_root, "run1_sub0001_first.vhdr")
    touch_recording(fastr_root, "run1_sub0001_second.vhdr")
    touch_recording(analyzer_input_root, "run1_sub0001_input.vhdr")
    touch_recording(analyzer_output_root, "run1_sub0001_output.vhdr")

    with pytest.raises(BenchmarkInputError, match="duplicate"):
        discover_recording_pairs(
            fastr_root,
            analyzer_input_root,
            analyzer_output_root,
        )


def test_discover_recording_pairs_rejects_missing_reference(tmp_path: Path) -> None:
    fastr_root = tmp_path / "fastr"
    analyzer_input_root = tmp_path / "analyzer_input"
    analyzer_output_root = tmp_path / "analyzer_output"
    touch_recording(fastr_root, "run1_sub0001_source.vhdr")
    touch_recording(analyzer_input_root, "run2_sub0001_input.vhdr")
    touch_recording(analyzer_output_root, "run2_sub0001_output.vhdr")

    with pytest.raises(BenchmarkInputError, match="missing"):
        discover_recording_pairs(
            fastr_root,
            analyzer_input_root,
            analyzer_output_root,
        )


def test_discover_recording_pairs_rejects_step_three_as_fastr_input(
    tmp_path: Path,
) -> None:
    fastr_root = tmp_path / "step3_bcg_corrected"
    analyzer_input_root = tmp_path / "analyzer_input"
    analyzer_output_root = tmp_path / "analyzer_output"
    touch_recording(fastr_root, "run1_sub0001_source.vhdr")
    touch_recording(analyzer_input_root, "run1_sub0001_input.vhdr")
    touch_recording(analyzer_output_root, "run1_sub0001_output.vhdr")

    with pytest.raises(BenchmarkInputError, match="FASTR-only"):
        discover_recording_pairs(
            fastr_root,
            analyzer_input_root,
            analyzer_output_root,
        )


def test_validate_pair_geometry_requires_matching_ecg_samples() -> None:
    time = np.arange(10_000, dtype=float)
    fastr = _RawSnapshot(
        data_volts=np.vstack((time, np.sin(time / 10.0))),
        channel_names=("EEG 001", "ECG"),
        channel_types=("eeg", "ecg"),
        sampling_rate_hz=1_000.0,
    )
    analyzer_data = fastr.data_volts.copy()
    analyzer_data[1] *= -1.0
    analyzer_input = _RawSnapshot(
        data_volts=analyzer_data,
        channel_names=fastr.channel_names,
        channel_types=fastr.channel_types,
        sampling_rate_hz=fastr.sampling_rate_hz,
    )
    analyzer_output = analyzer_input

    with pytest.raises(BenchmarkInputError, match="insufficient interior correlation"):
        _validate_pair_geometry(
            fastr,
            analyzer_input,
            analyzer_output,
            ecg_index=1,
            comparison_band_hz=(1.0, 100.0),
        )


def test_validate_pair_geometry_allows_one_fastr_endpoint_sample() -> None:
    time = np.arange(10_000, dtype=float)
    fastr = _RawSnapshot(
        data_volts=np.vstack((time, np.sin(time / 10.0))),
        channel_names=("EEG 001", "ECG"),
        channel_types=("eeg", "ecg"),
        sampling_rate_hz=1_000.0,
    )
    analyzer_input = _RawSnapshot(
        data_volts=fastr.data_volts[:, :-1],
        channel_names=fastr.channel_names,
        channel_types=fastr.channel_types,
        sampling_rate_hz=fastr.sampling_rate_hz,
    )

    correlation, comparison_count = _validate_pair_geometry(
        fastr,
        analyzer_input,
        analyzer_input,
        ecg_index=1,
        comparison_band_hz=(1.0, 100.0),
    )

    assert comparison_count == 9_999
    assert correlation == pytest.approx(1.0)
