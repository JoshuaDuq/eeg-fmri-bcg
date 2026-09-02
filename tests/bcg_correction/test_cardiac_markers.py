from pathlib import Path

import mne
import numpy as np
import pytest
from pybv import write_brainvision

from bcg_correction.bcg_config import DetectionRunConfig, DetectorConfig
from bcg_correction.brainvision import (
    BrainVisionMarker,
    read_brainvision_markers,
    write_brainvision_markers,
)
from bcg_correction.cardiac import CardiacInputError
from bcg_correction.cardiac_markers import (
    PULSE_MARKER_DESCRIPTION,
    PULSE_MARKER_TYPE,
    CardiacMarkerError,
    append_pulse_markers,
    audit_marker_trains,
    write_marker_recording,
)
from bcg_correction.cardiac_pipeline import run_cardiac_detection


def make_source_markers() -> tuple[BrainVisionMarker, ...]:
    return (
        BrainVisionMarker("New Segment", "", 1, 1, 0),
        BrainVisionMarker("Comment", "keep me", 201, 10, 2),
    )


def test_append_pulse_markers_preserves_source_markers() -> None:
    appended = append_pulse_markers(
        make_source_markers(),
        np.array([9, 409], dtype=np.int64),
        sample_count=1_000,
    )

    assert appended == (
        BrainVisionMarker("New Segment", "", 1, 1, 0),
        BrainVisionMarker("Comment", "keep me", 201, 10, 2),
        BrainVisionMarker(PULSE_MARKER_TYPE, PULSE_MARKER_DESCRIPTION, 10, 1, 0),
        BrainVisionMarker(PULSE_MARKER_TYPE, PULSE_MARKER_DESCRIPTION, 410, 1, 0),
    )


def test_append_pulse_markers_rejects_existing_pulse_markers() -> None:
    with pytest.raises(CardiacMarkerError, match="must not contain"):
        append_pulse_markers(
            (
                *make_source_markers(),
                BrainVisionMarker("Pulse Artifact", "R", 101, 1, 1),
            ),
            np.array([9, 409], dtype=np.int64),
            sample_count=1_000,
        )


@pytest.mark.parametrize(
    "peak_samples",
    [
        np.array([-1, 10], dtype=np.int64),
        np.array([10, 1_000], dtype=np.int64),
        np.array([10, 10], dtype=np.int64),
    ],
)
def test_append_pulse_markers_rejects_invalid_peak_samples(
    peak_samples: np.ndarray,
) -> None:
    with pytest.raises(CardiacMarkerError):
        append_pulse_markers(
            make_source_markers(),
            peak_samples,
            sample_count=1_000,
        )


def test_audit_marker_trains_is_one_to_one_and_reports_signed_lags() -> None:
    audit = audit_marker_trains(
        np.array([300, 100, 200], dtype=np.int64),
        np.array([102, 198, 302], dtype=np.int64),
        tolerance_samples=5,
    )

    np.testing.assert_array_equal(audit.analyzer_samples, [100, 200, 300])
    np.testing.assert_array_equal(audit.detected_samples, [102, 198, 302])
    assert audit.matched_count == 3
    assert audit.tolerance_samples == 5
    assert audit.median_lag_samples == 2.0
    assert audit.lag_iqr_samples == 2.0


def test_audit_marker_trains_does_not_reuse_analyzer_marker() -> None:
    audit = audit_marker_trains(
        np.array([100, 200], dtype=np.int64),
        np.array([104, 108], dtype=np.int64),
        tolerance_samples=10,
    )

    assert audit.matched_count == 1


def test_audit_marker_trains_returns_explicit_empty_result() -> None:
    audit = audit_marker_trains(
        np.array([], dtype=np.int64),
        np.array([], dtype=np.int64),
        tolerance_samples=5,
    )

    assert audit.matched_count == 0
    assert audit.median_lag_samples is None
    assert audit.lag_iqr_samples is None


def make_source_recording(tmp_path: Path) -> tuple[Path, bytes]:
    sampling_rate_hz = 1_000.0
    samples = np.arange(6_000, dtype=float)
    ecg = np.zeros(samples.size, dtype=float)
    peak_samples = np.array([800, 1_650, 2_530, 3_440, 4_370, 5_310])
    for index, peak in enumerate(peak_samples):
        sign = -1.0 if index == 4 else 1.0
        ecg += sign * 1e-3 * np.exp(-0.5 * ((samples - peak) / 8.0) ** 2)
        ecg += 0.65e-3 * np.exp(
            -0.5 * ((samples - peak - 280.0) / 35.0) ** 2
        )
    data = np.vstack((np.zeros(samples.size), ecg))
    vhdr_path = tmp_path / "source.vhdr"
    write_brainvision(
        data=data,
        sfreq=sampling_rate_hz,
        ch_names=["EEG 001", "ECG"],
        fname_base="source",
        folder_out=tmp_path,
        events=[],
        unit="µV",
    )
    marker_path = vhdr_path.with_suffix(".vmrk")
    marker_path.unlink()
    write_brainvision_markers(
        marker_path,
        "source.eeg",
        make_source_markers(),
    )
    return vhdr_path, vhdr_path.with_suffix(".eeg").read_bytes()


def test_write_marker_recording_copies_binary_data_and_rewrites_references(
    tmp_path: Path,
) -> None:
    source_vhdr, source_eeg_bytes = make_source_recording(tmp_path)
    output_vhdr = tmp_path / "output.vhdr"

    write_marker_recording(
        source_vhdr,
        output_vhdr,
        peak_samples=np.array([800, 1_650, 2_530, 3_440, 4_370, 5_310]),
    )

    assert output_vhdr.with_suffix(".eeg").read_bytes() == source_eeg_bytes
    data_file_name, markers = read_brainvision_markers(
        output_vhdr.with_suffix(".vmrk")
    )
    assert data_file_name == "output.eeg"
    assert sum(
        marker.marker_type == PULSE_MARKER_TYPE
        and marker.description == PULSE_MARKER_DESCRIPTION
        for marker in markers
    ) == 6
    header = output_vhdr.read_text(encoding="utf-8")
    assert "DataFile=output.eeg" in header
    assert "MarkerFile=output.vmrk" in header
    raw = mne.io.read_raw_brainvision(output_vhdr, preload=False, verbose="ERROR")
    assert raw.n_times == 6_000


def test_write_marker_recording_refuses_existing_output_sidecar(
    tmp_path: Path,
) -> None:
    source_vhdr, _ = make_source_recording(tmp_path)
    output_vhdr = tmp_path / "output.vhdr"
    output_vhdr.with_suffix(".vmrk").write_text("occupied", encoding="utf-8")

    with pytest.raises(FileExistsError):
        write_marker_recording(
            source_vhdr,
            output_vhdr,
            peak_samples=np.array([800, 1_650, 2_530]),
        )


def test_run_cardiac_detection_writes_provenance_and_independent_markers(
    tmp_path: Path,
) -> None:
    source_vhdr, _ = make_source_recording(tmp_path)
    output_vhdr = tmp_path / "detected.vhdr"
    config = DetectionRunConfig(
        input_vhdr=source_vhdr,
        output_vhdr=output_vhdr,
        detector=DetectorConfig(
            ecg_channel="ECG",
            preprocessing_band_hz=(7.0, 40.0),
            teager_emphasis_hz=10.0,
            teager_smoothing_seconds=0.028,
            template_window_seconds=(-0.2, 0.4),
            minimum_rr_seconds=0.4,
            maximum_rr_seconds=1.5,
            candidate_refractory_seconds=0.25,
            candidate_prominence_mad=3.0,
            correlation_threshold=0.5,
            refinement_iterations=2,
        ),
    )

    summary = run_cardiac_detection(config)

    assert summary.output_vhdr == output_vhdr.resolve()
    assert summary.marker_count == 5
    assert summary.status == "degraded"
    assert summary.provenance_json.is_file()
    _, markers = read_brainvision_markers(output_vhdr.with_suffix(".vmrk"))
    detected = [
        marker.position - 1
        for marker in markers
        if marker.marker_type == PULSE_MARKER_TYPE
        and marker.description == PULSE_MARKER_DESCRIPTION
    ]
    np.testing.assert_allclose(detected, [800, 1_650, 2_530, 3_440, 5_310])


def test_run_cardiac_detection_rejects_missing_ecg_channel(
    tmp_path: Path,
) -> None:
    source_vhdr, _ = make_source_recording(tmp_path)
    config = DetectionRunConfig(
        input_vhdr=source_vhdr,
        output_vhdr=tmp_path / "missing-ecg.vhdr",
        detector=DetectorConfig(
            ecg_channel="MISSING",
            preprocessing_band_hz=(7.0, 40.0),
            teager_emphasis_hz=10.0,
            teager_smoothing_seconds=0.028,
            template_window_seconds=(-0.2, 0.4),
            minimum_rr_seconds=0.4,
            maximum_rr_seconds=1.5,
            candidate_refractory_seconds=0.25,
            candidate_prominence_mad=3.0,
            correlation_threshold=0.5,
            refinement_iterations=2,
        ),
    )

    with pytest.raises(CardiacInputError, match="configured ECG channel"):
        run_cardiac_detection(config)
