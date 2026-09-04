import json
from dataclasses import replace
from pathlib import Path

import mne
import numpy as np
import pytest
import yaml
from pybv import write_brainvision

import bcg_correction.bcg_pipeline as bcg_pipeline_module
from bcg_correction.bcg_config import load_correction_config
from bcg_correction.bcg_pipeline import run_bcg_correction
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
from bcgstudy.cli import main


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


def _write_long_recording(tmp_path: Path, *, n_beats: int = 14) -> Path:
    """A recording with enough beats for the report's 8-epoch minimum."""
    sampling_rate_hz = 1_000.0
    peak_samples = np.array([800 + index * 900 for index in range(n_beats)])
    sample_count = int(peak_samples[-1] + 1_500)
    samples = np.arange(sample_count, dtype=float)
    ecg = np.zeros(sample_count, dtype=float)
    eeg = 2e-6 + 2e-7 * np.sin(2.0 * np.pi * samples / 173.0)
    artifact = np.exp(-0.5 * ((samples - 15.0) / 18.0) ** 2)
    for peak in peak_samples:
        ecg += 1e-3 * np.exp(-0.5 * ((samples - peak) / 8.0) ** 2)
        eeg += 25e-6 * np.roll(artifact, int(peak + 210 - 15))
    write_brainvision(
        data=np.vstack((eeg, ecg)),
        sfreq=sampling_rate_hz,
        ch_names=["O1", "ECG"],
        fname_base="long",
        folder_out=tmp_path,
        events=[],
        unit="µV",
    )
    marker_path = tmp_path / "long.vmrk"
    marker_path.unlink()
    write_brainvision_markers(
        marker_path,
        "long.eeg",
        (
            BrainVisionMarker("New Segment", "", 1, 1, 0),
            BrainVisionMarker("Volume", "V  1", 101, 1, 0),
        ),
    )
    return tmp_path / "long.vhdr"


def _correction_yaml(tmp_path: Path, source_vhdr: Path, *, method: str = "aas") -> Path:
    document = {
        "input": {"vhdr": str(source_vhdr)},
        "output": {"vhdr": str(tmp_path / "corrected.vhdr")},
        "correction": {
            "method": method,
            "window_seconds": [-0.1, 0.2],
            "ecg_to_bcg_delay_seconds": 0.21,
            "aas_neighbor_count": 2,
            "pca_obs_components": 1,
            "evaluation": {
                "block_counts": [2, 5, 10, 20],
                "minimum_beats_per_block": 8,
            },
            "maximum_gap_fraction": 0.05,
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


def test_quality_gate_admits_an_rr_gap_as_the_only_degradation() -> None:
    """A gap leaves BCG uncorrected in a bounded span; it does not corrupt.

    No template is subtracted where no beat was detected, so the remedy is to
    mark that span bad and cap the total share -- not to refuse the recording.
    """
    detection = _degraded_detection("rr_above_maximum")

    bcg_pipeline_module._require_usable_detection(detection)


def test_quality_gate_admits_an_rr_gap_with_median_rr_outside_bounds() -> None:
    detection = _degraded_detection(
        "rr_above_maximum",
        median_rr_seconds=1.6,
        maximum_rr_seconds=1.8,
    )

    bcg_pipeline_module._require_usable_detection(detection)


def test_quality_gate_still_rejects_a_spurious_detection() -> None:
    """``rr_below_minimum`` means an extra beat, which injects artifact."""
    detection = _degraded_detection("rr_below_minimum")

    with pytest.raises(
        CardiacInputError,
        match=r"^degraded ECG detection: rr_below_minimum$",
    ):
        bcg_pipeline_module._require_usable_detection(detection)


def test_quality_gate_reports_only_the_fatal_reasons() -> None:
    detection = _degraded_detection(
        "rr_below_minimum",
        "rr_above_maximum",
        "low_prominence_candidate",
    )

    with pytest.raises(
        CardiacInputError,
        match=(
            r"^degraded ECG detection: "
            r"rr_below_minimum, low_prominence_candidate$"
        ),
    ):
        bcg_pipeline_module._require_usable_detection(detection)


def test_gap_burden_gate_admits_a_small_share() -> None:
    fraction = bcg_pipeline_module._require_tolerable_gaps(
        ((1_000, 1_200),),
        sample_count=10_000,
        maximum_gap_fraction=0.05,
    )

    assert fraction == pytest.approx(0.02)


def test_gap_burden_gate_refuses_a_dominant_share() -> None:
    with pytest.raises(
        CardiacInputError,
        match=r"RR gaps cover 30\.00% of the recording across 2 gap\(s\)",
    ):
        bcg_pipeline_module._require_tolerable_gaps(
            ((0, 2_000), (5_000, 6_000)),
            sample_count=10_000,
            maximum_gap_fraction=0.05,
        )


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


def test_gap_reason_without_a_real_gap_writes_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A claimed gap reason with no span past the bound proceeds cleanly."""
    source_vhdr = _write_recording(tmp_path)
    config = load_correction_config(_correction_yaml(tmp_path, source_vhdr))
    detection = _degraded_detection("rr_above_maximum")
    monkeypatch.setattr(
        bcg_pipeline_module,
        "detect_r_peaks",
        lambda *args, **kwargs: detection,
    )

    summary = run_bcg_correction(config)

    provenance = json.loads(summary.provenance_json.read_text())
    assert provenance["rr_gap_spans"] == []
    assert provenance["rr_gap_fraction"] == pytest.approx(0.0)


def test_tolerable_gap_writes_output_with_bad_bcg_markers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Under the cap the span is marked bad, not a reason to refuse."""
    source_vhdr = _write_recording(tmp_path)
    yaml_path = _correction_yaml(tmp_path, source_vhdr)
    document = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    document["correction"]["maximum_gap_fraction"] = 0.4
    yaml_path.write_text(yaml.safe_dump(document), encoding="utf-8")
    config = load_correction_config(yaml_path)
    detection = _degraded_detection(
        "rr_above_maximum",
        peak_samples=np.array([800, 1_650, 3_440, 4_370, 5_310], dtype=np.int64),
    )
    monkeypatch.setattr(
        bcg_pipeline_module,
        "detect_r_peaks",
        lambda *args, **kwargs: detection,
    )

    summary = run_bcg_correction(config)

    _, markers = read_brainvision_markers(summary.output_vhdr.with_suffix(".vmrk"))
    assert any(marker.description == "Bad_BCG" for marker in markers)
    provenance = json.loads(summary.provenance_json.read_text())
    assert provenance["rr_gap_spans"] == [[1_650, 3_440]]
    assert provenance["rr_gap_fraction"] == pytest.approx(1_790 / 6_000)
    assert provenance["maximum_gap_fraction"] == pytest.approx(0.4)


def test_dominant_gap_burden_writes_no_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Above the cap the detected beats are untrusted, so nothing is written."""
    source_vhdr = _write_recording(tmp_path)
    config = load_correction_config(_correction_yaml(tmp_path, source_vhdr))
    detection = _degraded_detection(
        "rr_above_maximum",
        peak_samples=np.array([800, 1_650, 3_440, 4_370, 5_310], dtype=np.int64),
    )
    monkeypatch.setattr(
        bcg_pipeline_module,
        "detect_r_peaks",
        lambda *args, **kwargs: detection,
    )

    with pytest.raises(CardiacInputError, match=r"RR gaps cover 29\.83%"):
        run_bcg_correction(config)

    assert all(
        not path.exists()
        for path in bcg_pipeline_module._output_paths(
            config.output_vhdr.expanduser().resolve()
        ).values()
    )


def test_residual_evaluation_is_not_an_output_acceptance_gate(tmp_path):
    source = _write_recording(tmp_path)
    config = load_correction_config(_correction_yaml(tmp_path, source))
    summary = run_bcg_correction(config)
    provenance = json.loads(summary.provenance_json.read_text())
    assert summary.output_vhdr.is_file()
    assert (
        provenance["residual_qc"]["status"] == "descriptive_only_not_an_acceptance_gate"
    )
    assert provenance["residual_qc"]["preservation_status"] == "not_measured"


def test_correction_writes_a_report_figure(tmp_path: Path) -> None:
    """Every corrected recording gets residual, spectra, and ratio pages."""
    from bcg_correction.correction_report import report_page_paths

    source_vhdr = _write_long_recording(tmp_path)
    config = load_correction_config(_correction_yaml(tmp_path, source_vhdr))

    summary = run_bcg_correction(config)

    report = bcg_pipeline_module._output_paths(summary.output_vhdr)["report"]
    pages = report_page_paths(report)
    assert pages.keys() == {"residual", "spectra", "ratios"}
    for page in pages.values():
        assert page.is_file()
        assert page.stat().st_size > 10_000


def test_report_metrics_match_the_provenance_ratio(tmp_path: Path) -> None:
    """The page must not disagree with the number written beside it."""
    source_vhdr = _write_long_recording(tmp_path)
    config = load_correction_config(_correction_yaml(tmp_path, source_vhdr))

    summary = run_bcg_correction(config)

    provenance = json.loads(summary.provenance_json.read_text())
    raw = mne.io.read_raw_brainvision(source_vhdr, preload=True, verbose="ERROR")
    corrected = mne.io.read_raw_brainvision(
        summary.output_vhdr, preload=True, verbose="ERROR"
    )
    profile = bcg_pipeline_module.compute_correction_profile(
        raw.get_data(),
        corrected.get_data(),
        tuple(raw.ch_names),
        ecg_channel_index=raw.ch_names.index("ECG"),
        peak_samples=np.asarray(provenance["peak_samples"], dtype=np.int64),
        sampling_rate_hz=float(raw.info["sfreq"]),
        delay_seconds=provenance["ecg_to_bcg_delay_seconds"],
        window_seconds=tuple(provenance["window_seconds"]),
        gap_fraction=provenance["rr_gap_fraction"],
        method=provenance["method"],
        evaluation=config.evaluation,
    )
    assert profile is not None
    assert profile.preservation_status == "not_measured"
    assert tuple(profile.block_counts) == config.evaluation.block_counts
    from bcg_correction.correction_report import profile_metrics

    for key, value in profile_metrics(profile).items():
        saved = provenance["residual_qc"]["metrics"][key]
        if isinstance(value, float):
            assert value == pytest.approx(saved, rel=2e-5, abs=1e-10)
        else:
            assert value == saved


def test_ok_detection_writes_no_bad_bcg_markers(tmp_path: Path) -> None:
    source_vhdr = _write_recording(tmp_path)
    config = load_correction_config(_correction_yaml(tmp_path, source_vhdr))

    summary = run_bcg_correction(config)

    _, markers = read_brainvision_markers(summary.output_vhdr.with_suffix(".vmrk"))
    assert all(marker.description != "Bad_BCG" for marker in markers)
    provenance = __import__("json").loads(summary.provenance_json.read_text())
    assert provenance["rr_gap_spans"] == []


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

    provenance = json.loads(summary.provenance_json.read_text(encoding="utf-8"))
    assert (
        provenance["residual_qc"]["status"] == "descriptive_only_not_an_acceptance_gate"
    )


def test_old_residual_acceptance_threshold_is_rejected(tmp_path):
    from bcg_correction.config import ConfigurationError

    source = _write_recording(tmp_path)
    path = _correction_yaml(tmp_path, source)
    document = yaml.safe_load(path.read_text())
    document["correction"]["maximum_residual_ratio"] = 0.5
    path.write_text(yaml.safe_dump(document))
    with pytest.raises(ConfigurationError, match="maximum_residual_ratio"):
        load_correction_config(path)


def test_correct_bcg_cli_executes_yaml_pipeline(tmp_path: Path, capsys) -> None:
    source_vhdr = _write_recording(tmp_path)
    config_path = _correction_yaml(tmp_path, source_vhdr)

    assert main(["correct", "--config", str(config_path)]) == 0

    summary = __import__("json").loads(capsys.readouterr().out)
    assert summary["method"] == "aas"
    assert summary["marker_count"] >= 4


def test_profile_round_trips_and_aggregates(tmp_path: Path) -> None:
    """Profiles persist losslessly and average into a subject/cohort page."""
    from bcg_correction.correction_report import (
        read_profile,
        save_aggregate_report,
        write_profile,
    )

    source_vhdr = _write_long_recording(tmp_path)
    config = load_correction_config(_correction_yaml(tmp_path, source_vhdr))
    summary = run_bcg_correction(config)
    stored = bcg_pipeline_module._output_paths(summary.output_vhdr)["profile"]
    assert stored.is_file()

    profile = read_profile(stored)
    assert profile.preservation_status == "not_measured"
    assert tuple(profile.block_counts) == config.evaluation.block_counts
    round_trip = tmp_path / "again.npz"
    write_profile(profile, round_trip)
    again = read_profile(round_trip)
    np.testing.assert_allclose(again.local_ratio, profile.local_ratio)

    page = tmp_path / "aggregate.png"
    from dataclasses import replace

    from bcg_correction.correction_report import report_page_paths

    assert save_aggregate_report(
        [profile, replace(profile, label="second")], title="two", output=page
    )
    pages = report_page_paths(page)
    assert all(path.stat().st_size > 10_000 for path in pages.values())


def test_profile_without_schema_version_is_rejected(tmp_path: Path) -> None:
    from bcg_correction.correction_report import read_profile

    path = tmp_path / "stale_profile.npz"
    np.savez_compressed(path, method=np.asarray("aas"))

    with pytest.raises(ValueError, match="schema_version"):
        read_profile(path)


def test_aggregate_report_declines_an_empty_list(tmp_path: Path) -> None:
    from bcg_correction.correction_report import save_aggregate_report

    assert not save_aggregate_report([], title="none", output=tmp_path / "x.png")


def test_provenance_records_every_method_parameter(tmp_path: Path) -> None:
    """A correction that cannot be reproduced from its own sidecar is not
    provenance. Every arm's shape parameter has to be recorded."""
    source_vhdr = _write_recording(tmp_path)
    config = load_correction_config(
        _correction_yaml(tmp_path, source_vhdr, method="aas")
    )

    summary = run_bcg_correction(config)

    provenance = json.loads(summary.provenance_json.read_text())
    assert provenance["method"] == "aas"
    for key in ("aas_neighbor_count", "pca_obs_components"):
        assert key in provenance, key
