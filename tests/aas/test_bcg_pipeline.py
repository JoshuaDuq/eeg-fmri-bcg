import json
from dataclasses import replace
from pathlib import Path

import mne
import numpy as np
import pytest
import yaml
from pybv import write_brainvision

import bcg_correction.bcg_pipeline as bcg_pipeline_module
from bcg_correction.bcg import BcgInputError
from bcg_correction.bcg_config import load_correction_config
from bcg_correction.bcg_pipeline import (
    _bcg_psd_interval,
    _prepare_bcg_psd_raw,
    run_bcg_correction,
)
from bcg_correction.brainvision import (
    BrainVisionMarker,
    read_brainvision_markers,
    write_brainvision_markers,
)
from bcg_correction.cardiac import (
    CardiacDetection,
    CardiacDetectionQuality,
    CardiacInputError,
)
from bcg_correction.cardiac_markers import (
    PULSE_MARKER_DESCRIPTION,
    PULSE_MARKER_TYPE,
)
from bcg_correction.cli import main


def _write_recording(tmp_path: Path, *, bcg_delay_samples: int = 210) -> Path:
    sampling_rate_hz = 1_000.0
    sample_count = 6_000
    samples = np.arange(sample_count, dtype=float)
    peak_samples = np.array([800, 1_650, 2_530, 3_440, 4_370, 5_310])
    ecg = np.zeros(sample_count, dtype=float)
    eeg = 2e-6 + 2e-7 * np.sin(2.0 * np.pi * samples / 173.0)
    artifact = np.exp(-0.5 * ((samples - 15.0) / 18.0) ** 2)
    for peak in peak_samples:
        centre = peak + bcg_delay_samples
        ecg += 1e-3 * np.exp(-0.5 * ((samples - peak) / 8.0) ** 2)
        eeg += 25e-6 * np.roll(artifact, int(centre - 15))
    write_brainvision(
        data=np.vstack((eeg, ecg)),
        sfreq=sampling_rate_hz,
        ch_names=["O1", "ECG"],
        fname_base="source",
        folder_out=tmp_path,
        events=[],
        unit="µV",
    )
    marker_path = tmp_path / "source.vmrk"
    marker_path.unlink()
    write_brainvision_markers(
        marker_path,
        "source.eeg",
        (
            BrainVisionMarker("New Segment", "", 1, 1, 0),
            BrainVisionMarker("Volume", "V  1", 101, 1, 0),
        ),
    )
    return tmp_path / "source.vhdr"


def _correction_yaml(tmp_path: Path, source_vhdr: Path) -> Path:
    document = {
        "input": {"vhdr": str(source_vhdr)},
        "output": {"vhdr": str(tmp_path / "corrected.vhdr")},
        "correction": {
            "method": "aas",
            "window_seconds": [-0.1, 0.2],
            "ecg_to_bcg_delay_seconds": 0.21,
            "aas_neighbor_count": 2,
            "pca_obs_components": 1,
            "maximum_residual_ratio": 0.75,
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
    path = tmp_path / "bcg.yml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path


def _degraded_detection(
    *degradation_reasons: str,
    median_rr_seconds: float = 0.91,
    maximum_rr_seconds: float = 1.6,
    peak_samples: np.ndarray | None = None,
) -> CardiacDetection:
    if peak_samples is None:
        peak_samples = np.array(
            [800, 1_650, 2_530, 3_440, 4_370, 5_310],
            dtype=np.int64,
        )
    quality = CardiacDetectionQuality(
        candidate_count=6,
        selected_polarity=1,
        positive_candidate_count=6,
        negative_candidate_count=0,
        accepted_count=int(peak_samples.size),
        rejected_count=0,
        median_rr_seconds=median_rr_seconds,
        rr_iqr_seconds=0.06,
        minimum_rr_seconds=0.85,
        maximum_rr_seconds=maximum_rr_seconds,
        implied_rate_bpm=65.93,
        template_correlation_median=0.95,
        rejected_low_prominence=0,
        rejected_low_correlation=0,
        rejected_double_mark=0,
        rejected_interval=0,
        degradation_reasons=degradation_reasons,
        status="degraded",
    )
    return CardiacDetection(peak_samples=peak_samples, quality=quality)


def test_quality_gate_rejects_degraded_detection_with_median_in_bounds() -> None:
    detection = _degraded_detection("rr_above_maximum")

    with pytest.raises(
        CardiacInputError,
        match=r"^degraded ECG detection: rr_above_maximum$",
    ):
        bcg_pipeline_module._require_usable_detection(
            detection,
        )


def test_quality_gate_rejects_median_rr_outside_bounds() -> None:
    detection = _degraded_detection(
        "rr_above_maximum",
        median_rr_seconds=1.6,
        maximum_rr_seconds=1.8,
    )

    with pytest.raises(
        CardiacInputError,
        match=r"^degraded ECG detection: rr_above_maximum$",
    ):
        bcg_pipeline_module._require_usable_detection(detection)


def test_quality_gate_uses_status_when_reasons_are_empty() -> None:
    detection = _degraded_detection()

    with pytest.raises(
        CardiacInputError,
        match=r"^degraded ECG detection$",
    ):
        bcg_pipeline_module._require_usable_detection(detection)


def test_quality_gate_rejects_inconsistent_ok_status() -> None:
    detection = _degraded_detection("rr_above_maximum")
    inconsistent = replace(
        detection,
        quality=replace(detection.quality, status="ok"),
    )

    with pytest.raises(
        CardiacInputError,
        match=r"^inconsistent ECG detection quality$",
    ):
        bcg_pipeline_module._require_usable_detection(
            inconsistent,
        )


def test_degraded_detection_with_median_in_bounds_writes_no_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_vhdr = _write_recording(tmp_path)
    config = load_correction_config(_correction_yaml(tmp_path, source_vhdr))
    peak_samples = np.array([800, 1_700, 2_600, 4_600, 5_500], dtype=np.int64)
    detection = _degraded_detection(
        "rr_above_maximum",
        median_rr_seconds=0.9,
        maximum_rr_seconds=2.0,
        peak_samples=peak_samples,
    )
    monkeypatch.setattr(
        bcg_pipeline_module,
        "detect_r_peaks",
        lambda *args, **kwargs: detection,
    )

    with pytest.raises(
        CardiacInputError,
        match=r"^degraded ECG detection: rr_above_maximum$",
    ):
        run_bcg_correction(config)

    assert all(not path.exists() for path in bcg_pipeline_module._output_paths(
        config.output_vhdr.expanduser().resolve()
    ).values())


def test_unusable_median_rr_writes_no_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_vhdr = _write_recording(tmp_path)
    config = load_correction_config(_correction_yaml(tmp_path, source_vhdr))
    detection = _degraded_detection(
        "rr_above_maximum",
        median_rr_seconds=1.6,
        maximum_rr_seconds=1.8,
    )
    monkeypatch.setattr(
        bcg_pipeline_module,
        "detect_r_peaks",
        lambda *args, **kwargs: detection,
    )

    with pytest.raises(
        CardiacInputError,
        match=r"^degraded ECG detection: rr_above_maximum$",
    ):
        run_bcg_correction(config)

    output_vhdr = config.output_vhdr.expanduser().resolve()
    output_paths = (
        output_vhdr,
        output_vhdr.with_suffix(".eeg"),
        output_vhdr.with_suffix(".vmrk"),
        output_vhdr.with_suffix(".bcg.json"),
    )
    assert all(not path.exists() for path in output_paths)


def test_pipeline_applies_estimated_delay_not_yaml_delay(tmp_path: Path) -> None:
    source_vhdr = _write_recording(tmp_path, bcg_delay_samples=80)
    config = load_correction_config(_correction_yaml(tmp_path, source_vhdr))

    summary = run_bcg_correction(config)

    provenance = __import__("json").loads(summary.provenance_json.read_text())
    assert provenance["ecg_to_bcg_delay_seconds"] == pytest.approx(0.08, abs=0.01)
    assert provenance["delay_estimation"]["configured_delay_seconds"] == pytest.approx(
        0.21
    )
    assert provenance["delay_estimation"]["best_delay_seconds"] == pytest.approx(
        0.08,
        abs=0.01,
    )
    _, markers = read_brainvision_markers(summary.output_vhdr.with_suffix(".vmrk"))
    pulse = [
        marker
        for marker in markers
        if marker.marker_type == PULSE_MARKER_TYPE
        and marker.description == PULSE_MARKER_DESCRIPTION
    ]
    assert abs(pulse[0].position - 801) <= 20


def test_ok_detection_writes_no_bad_bcg_markers(tmp_path: Path) -> None:
    source_vhdr = _write_recording(tmp_path)
    config = load_correction_config(_correction_yaml(tmp_path, source_vhdr))

    summary = run_bcg_correction(config)

    _, markers = read_brainvision_markers(summary.output_vhdr.with_suffix(".vmrk"))
    assert all(marker.description != "Bad_BCG" for marker in markers)
    provenance = __import__("json").loads(summary.provenance_json.read_text())
    assert provenance["rr_gap_spans"] == []


def test_bcg_correction_refuses_existing_psd_output(tmp_path: Path) -> None:
    source_vhdr = _write_recording(tmp_path)
    config = load_correction_config(_correction_yaml(tmp_path, source_vhdr))
    existing_psd = tmp_path / "corrected_psd_before.png"
    existing_psd.write_bytes(b"occupied")

    with pytest.raises(FileExistsError, match=r"corrected_psd_before\.png"):
        run_bcg_correction(config)

    assert not config.output_vhdr.exists()


def test_run_bcg_correction_preserves_ecg_and_writes_pulse_markers(
    tmp_path: Path,
) -> None:
    source_vhdr = _write_recording(tmp_path)
    config = load_correction_config(_correction_yaml(tmp_path, source_vhdr))

    summary = run_bcg_correction(config)

    assert summary.method == "aas"
    assert summary.marker_count >= 4
    assert summary.output_vhdr.is_file()
    assert summary.provenance_json.is_file()
    source = mne.io.read_raw_brainvision(source_vhdr, preload=True, verbose="ERROR")
    corrected = mne.io.read_raw_brainvision(
        summary.output_vhdr, preload=True, verbose="ERROR"
    )
    np.testing.assert_allclose(
        corrected.get_data(picks=["ECG"]),
        source.get_data(picks=["ECG"]),
        atol=1e-10,
        rtol=0.0,
    )
    _, markers = read_brainvision_markers(summary.output_vhdr.with_suffix(".vmrk"))
    pulse_count = sum(
        marker.marker_type == PULSE_MARKER_TYPE
        and marker.description == PULSE_MARKER_DESCRIPTION
        for marker in markers
    )
    assert pulse_count == summary.marker_count

    assert summary.psd_before == tmp_path / "corrected_psd_before.png"
    assert summary.psd_after == tmp_path / "corrected_psd_after.png"
    assert summary.psd_before.is_file()
    assert summary.psd_after.is_file()
    provenance = json.loads(summary.provenance_json.read_text(encoding="utf-8"))
    assert provenance["psd_before"] == str(summary.psd_before)
    assert provenance["psd_after"] == str(summary.psd_after)
    assert provenance["residual_qc"]["ratio"] <= 0.75
    assert 0.0 <= provenance["psd_interval_seconds"]["start"]
    assert provenance["psd_interval_seconds"]["start"] < provenance[
        "psd_interval_seconds"
    ]["end"]


def test_bcg_correction_refuses_excessive_residual_before_writing(
    tmp_path: Path,
) -> None:
    source_vhdr = _write_recording(tmp_path)
    config_path = _correction_yaml(tmp_path, source_vhdr)
    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    document["correction"]["maximum_residual_ratio"] = 1e-6
    config_path.write_text(yaml.safe_dump(document), encoding="utf-8")
    config = load_correction_config(config_path)

    with pytest.raises(BcgInputError, match="residual ratio"):
        run_bcg_correction(config)

    output_paths = bcg_pipeline_module._output_paths(
        config.output_vhdr.expanduser().resolve()
    )
    assert all(not path.exists() for path in output_paths.values())


def test_bcg_psd_removes_only_historical_gradient_annotations(
    tmp_path: Path,
) -> None:
    source_vhdr = _write_recording(tmp_path)
    marker_path = source_vhdr.with_suffix(".vmrk")
    _, markers = read_brainvision_markers(marker_path)
    marker_path.unlink()
    write_brainvision_markers(
        marker_path,
        "source.eeg",
        (
            *markers,
            BrainVisionMarker("Bad Interval", "Bad_Gradient", 1, 500, 0),
            BrainVisionMarker("Bad Interval", "Bad_manual", 1_001, 100, 0),
        ),
    )
    raw = mne.io.read_raw_brainvision(
        source_vhdr,
        preload=False,
        verbose="ERROR",
    )
    prepared = _prepare_bcg_psd_raw(raw)
    try:
        descriptions = set(prepared.annotations.description)
        assert "Bad Interval/Bad_Gradient" not in descriptions
        assert "Bad Interval/Bad_manual" in descriptions
    finally:
        prepared.close()
        raw.close()


def test_bcg_psd_interval_uses_longest_contiguous_corrected_window() -> None:
    windows = np.concatenate(
        [
            np.arange(100, 400),
            np.arange(500, 800),
            np.arange(900, 1_200),
            np.arange(1_300, 1_600),
        ]
    )

    tmin, tmax, n_fft = _bcg_psd_interval(
        windows,
        sampling_rate_hz=1_000.0,
        sample_count=2_000,
    )

    assert tmin == 0.1
    assert tmax == 0.4
    assert n_fft == 300


def test_bcg_psd_interval_caps_longest_contiguous_window_at_2048_samples() -> None:
    samples = np.concatenate([np.arange(0, 300), np.arange(400, 5_000)])

    tmin, tmax, n_fft = _bcg_psd_interval(
        samples,
        sampling_rate_hz=1_000.0,
        sample_count=5_000,
    )

    assert tmin == 0.4
    assert tmax == 5.0
    assert n_fft == 2_048


def test_bcg_psd_diagnostics_share_the_corrected_interval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_vhdr = _write_recording(tmp_path)
    config = load_correction_config(_correction_yaml(tmp_path, source_vhdr))
    calls: list[tuple[mne.io.BaseRaw, float, float, float, int | None]] = []

    def capture_plot(
        raw: mne.io.BaseRaw,
        output_path: Path,
        *,
        fmax: float,
        title: str,
        tmin: float,
        tmax: float,
        n_fft: int | None = None,
    ) -> None:
        calls.append((raw, fmax, tmin, tmax, n_fft))
        output_path.touch()

    monkeypatch.setattr(
        bcg_pipeline_module,
        "save_psd_plot",
        capture_plot,
        raising=False,
    )

    run_bcg_correction(config)

    assert len(calls) == 2
    assert calls[0][1:] == calls[1][1:]
    assert calls[0][1] == 100.0
    assert calls[0][2] < calls[0][3]
    assert calls[0][4] == 300
    assert all(
        "bad_psd_gap" not in raw.annotations.description
        for raw, _, _, _, _ in calls
    )


def test_correct_bcg_cli_executes_yaml_pipeline(tmp_path: Path, capsys) -> None:
    source_vhdr = _write_recording(tmp_path)
    config_path = _correction_yaml(tmp_path, source_vhdr)

    assert main(["correct-bcg", "--config", str(config_path)]) == 0

    summary = __import__("json").loads(capsys.readouterr().out)
    assert summary["method"] == "aas"
    assert summary["marker_count"] >= 4
