"""Paired benchmark orchestration for independent BCG correction methods."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import mne
import numpy as np
from scipy.signal import butter, sosfiltfilt

from .bcg import BcgCorrectionConfig, BcgInputError, correct_bcg
from .bcg_config import BenchmarkConfig
from .brainvision import BrainVisionMarkerError
from .brainvision_io import (
    BrainVisionInputError,
    BrainVisionRecording,
    read_brainvision_recording,
    select_marker_samples,
    write_brainvision_recording,
)
from .cardiac import CardiacInputError, detect_r_peaks
from .cardiac_markers import (
    CardiacMarkerError,
    append_pulse_markers,
    audit_marker_trains,
    validate_fastr_marker_input,
)
from .metrics import (
    MetricInputError,
    cardiac_residual_ratio,
    circular_shifted_cardiac_null,
    delay_estimation_eeg,
    estimate_ecg_to_bcg_delay,
    held_out_cardiac_rms,
)


class BenchmarkInputError(ValueError):
    """Raised when a benchmark cannot form a valid recording pair."""


@dataclass(frozen=True, slots=True)
class RecordingPair:
    """One FASTR input and the corresponding Analyzer before/after pair."""

    recording_id: str
    fastr_vhdr: Path
    analyzer_input_vhdr: Path
    analyzer_output_vhdr: Path


@dataclass(frozen=True, slots=True)
class BenchmarkSummary:
    """Report paths and run counts for one paired benchmark."""

    report_json: Path
    report_csv: Path
    run_count: int
    successful_count: int
    failed_count: int


@dataclass(frozen=True, slots=True)
class _RawSnapshot:
    data_volts: np.ndarray
    channel_names: tuple[str, ...]
    channel_types: tuple[str, ...]
    sampling_rate_hz: float


_RUN_KEY_PATTERN = re.compile(
    r"(?P<run>(?:run(?P<run_number>\d+)|baseline))"
    r".*?sub[-_]?(?P<subject>[A-Za-z0-9]+)",
    re.IGNORECASE,
)
_EXPECTED_RUN_ERRORS = (
    OSError,
    RuntimeError,
    BenchmarkInputError,
    BrainVisionInputError,
    BrainVisionMarkerError,
    BcgInputError,
    CardiacInputError,
    CardiacMarkerError,
    MetricInputError,
)


def _recording_id(path: Path) -> str:
    match = _RUN_KEY_PATTERN.search(path.stem)
    if match is None:
        raise BenchmarkInputError(
            "cannot derive recording key from BrainVision filename: "
            f"{path.name}"
        )
    run = match.group("run").lower()
    if run == "baseline":
        run_id = "baseline"
    else:
        run_id = f"run{int(match.group('run_number'))}"
    subject = match.group("subject").lower()
    return f"{run_id}_sub{subject}"


def _index_recordings(root: Path, label: str) -> dict[str, Path]:
    if not root.is_dir():
        raise BenchmarkInputError(f"{label} root does not exist: {root}")
    paths = tuple(
        sorted(
            path
            for path in root.rglob("*.vhdr")
            if not path.name.startswith("._")
        )
    )
    if not paths:
        raise BenchmarkInputError(f"{label} root contains no .vhdr files: {root}")
    indexed: dict[str, Path] = {}
    for path in paths:
        recording_id = _recording_id(path)
        if recording_id in indexed:
            raise BenchmarkInputError(
                f"duplicate {label} recording key {recording_id!r}: "
                f"{indexed[recording_id]} and {path}"
            )
        indexed[recording_id] = path.resolve()
    return indexed


def discover_recording_pairs(
    fastr_root: str | Path,
    analyzer_input_root: str | Path,
    analyzer_output_root: str | Path,
) -> tuple[RecordingPair, ...]:
    """Discover strict three-way pairs by subject and run identity."""
    fastr_directory = Path(fastr_root).expanduser().resolve()
    analyzer_input_directory = (
        Path(analyzer_input_root).expanduser().resolve()
    )
    analyzer_output_directory = (
        Path(analyzer_output_root).expanduser().resolve()
    )
    if fastr_directory.name == "step3_bcg_corrected":
        raise BenchmarkInputError(
            "fastr_root must be the FASTR-only input, not step3_bcg_corrected"
        )
    directories = (
        fastr_directory,
        analyzer_input_directory,
        analyzer_output_directory,
    )
    if len(set(directories)) != len(directories):
        raise BenchmarkInputError("benchmark input roots must be different")
    fastr = _index_recordings(fastr_directory, "FASTR")
    analyzer_input = _index_recordings(
        analyzer_input_directory,
        "Analyzer input",
    )
    analyzer_output = _index_recordings(
        analyzer_output_directory,
        "Analyzer output",
    )
    fastr_keys = set(fastr)
    analyzer_input_keys = set(analyzer_input)
    analyzer_output_keys = set(analyzer_output)
    expected_keys = fastr_keys | analyzer_input_keys | analyzer_output_keys
    missing = []
    for label, keys in (
        ("FASTR", fastr_keys),
        ("Analyzer input", analyzer_input_keys),
        ("Analyzer output", analyzer_output_keys),
    ):
        absent = sorted(expected_keys - keys)
        if absent:
            missing.append(f"{label} missing " + ", ".join(absent))
    if missing:
        raise BenchmarkInputError("recording pairs are missing: " + "; ".join(missing))
    return tuple(
        RecordingPair(
            recording_id=recording_id,
            fastr_vhdr=fastr[recording_id],
            analyzer_input_vhdr=analyzer_input[recording_id],
            analyzer_output_vhdr=analyzer_output[recording_id],
        )
        for recording_id in sorted(expected_keys)
    )



def _benchmark_pair(pair: RecordingPair, config: BenchmarkConfig) -> dict[str, object]:
    fastr = _read_raw_snapshot(pair.fastr_vhdr)
    fastr_recording = read_brainvision_recording(pair.fastr_vhdr)
    validate_fastr_marker_input(fastr_recording.markers)
    ecg_index = _channel_index(fastr.channel_names, config.detector.ecg_channel)
    detection = detect_r_peaks(
        fastr.data_volts[ecg_index],
        fastr.sampling_rate_hz,
        config=config.detector,
    )

    analyzer_input = _read_raw_snapshot(pair.analyzer_input_vhdr)
    analyzer_output = _read_raw_snapshot(pair.analyzer_output_vhdr)
    analyzer_output_recording = read_brainvision_recording(
        pair.analyzer_output_vhdr
    )
    ecg_correlation, comparison_sample_count = _validate_pair_geometry(
        fastr,
        analyzer_input,
        analyzer_output,
        ecg_index,
        comparison_band_hz=config.detector.preprocessing_band_hz,
    )
    analyzer_input_recording = read_brainvision_recording(
        pair.analyzer_input_vhdr
    )
    analyzer_samples = select_marker_samples(
        analyzer_input_recording.markers,
        marker_type="Pulse Artifact",
        marker_description="R",
        sample_count=analyzer_input.data_volts.shape[1],
    )
    tolerance_samples = round(
        config.marker_tolerance_seconds * fastr.sampling_rate_hz
    )
    marker_audit = audit_marker_trains(
        analyzer_samples,
        detection.peak_samples,
        tolerance_samples=tolerance_samples,
    )
    fastr_data = fastr.data_volts
    fastr_metric_data = fastr_data[:, :comparison_sample_count]
    analyzer_input_data = analyzer_input.data_volts[:, :comparison_sample_count]
    analyzer_output_data = analyzer_output.data_volts[:, :comparison_sample_count]
    eeg_picks = _eeg_picks(fastr.channel_types, ecg_index)
    delay_scan = estimate_ecg_to_bcg_delay(
        delay_estimation_eeg(
            fastr_data,
            fastr.channel_names,
            ecg_channel_index=ecg_index,
        ),
        detection.peak_samples,
        sampling_rate_hz=fastr.sampling_rate_hz,
    )
    applied_delay = delay_scan.best_delay_seconds
    correction_peak_samples = _complete_event_samples(
        detection.peak_samples,
        fastr.sampling_rate_hz,
        applied_delay,
        comparison_sample_count,
        config.correction_window_seconds,
    )
    artifact_anchors = _artifact_anchors(
        correction_peak_samples,
        fastr.sampling_rate_hz,
        applied_delay,
        comparison_sample_count,
    )
    analyzer_metric_samples = _complete_event_samples(
        analyzer_samples,
        analyzer_input.sampling_rate_hz,
        config.ecg_to_bcg_delay_seconds,
        comparison_sample_count,
        config.correction_window_seconds,
    )
    analyzer_artifact_anchors = _artifact_anchors(
        analyzer_metric_samples,
        analyzer_input.sampling_rate_hz,
        config.ecg_to_bcg_delay_seconds,
        comparison_sample_count,
    )
    fastr_eeg_uv = fastr_metric_data[eeg_picks] * 1e6
    analyzer_eeg_uv = analyzer_input_data[eeg_picks] * 1e6
    fastr_null_values = circular_shifted_cardiac_null(
        fastr_eeg_uv,
        artifact_anchors,
        sampling_rate_hz=fastr.sampling_rate_hz,
        window_seconds=config.correction_window_seconds,
        surrogate_count=config.null_surrogate_count,
        seed=config.random_seed,
    )
    analyzer_null_values = circular_shifted_cardiac_null(
        analyzer_eeg_uv,
        analyzer_artifact_anchors,
        sampling_rate_hz=analyzer_input.sampling_rate_hz,
        window_seconds=config.correction_window_seconds,
        surrogate_count=config.null_surrogate_count,
        seed=config.random_seed,
    )
    method_rows: dict[str, object] = {}
    method_rows["analyzer_reference"] = _method_metrics(
        analyzer_input_data,
        analyzer_output_data,
        eeg_picks,
        ecg_index,
        analyzer_artifact_anchors,
        analyzer_input.sampling_rate_hz,
        config.correction_window_seconds,
        analyzer_null_values,
        corrected_samples=None,
    )
    for method in config.correction_methods:
        correction_config = BcgCorrectionConfig(
            method=method,
            window_seconds=config.correction_window_seconds,
            ecg_to_bcg_delay_seconds=applied_delay,
            aas_neighbor_count=config.aas_neighbor_count,
            pca_obs_components=config.pca_obs_components,
        )
        result = correct_bcg(
            fastr_data,
            correction_peak_samples,
            fastr.sampling_rate_hz,
            channel_names=fastr.channel_names,
            eeg_picks=eeg_picks,
            ecg_channel_index=ecg_index,
            config=correction_config,
        )
        output_vhdr = (
            config.output_root
            / pair.recording_id
            / method
            / f"{pair.recording_id}_{method}.vhdr"
        )
        output_markers = append_pulse_markers(
            fastr_recording.markers,
            detection.peak_samples,
            sample_count=fastr_data.shape[1],
        )
        write_brainvision_recording(
            data=result.data_volts,
            sampling_rate=fastr.sampling_rate_hz,
            channel_names=fastr.channel_names,
            output_vhdr=output_vhdr,
            markers=output_markers,
        )
        method_rows[method] = _method_metrics(
            fastr_metric_data,
            result.data_volts[:, :comparison_sample_count],
            eeg_picks,
            ecg_index,
            artifact_anchors,
            fastr.sampling_rate_hz,
            config.correction_window_seconds,
            fastr_null_values,
            corrected_samples=result.corrected_samples[
                result.corrected_samples < comparison_sample_count
            ],
        ) | {"output_vhdr": str(output_vhdr.resolve())}
    return {
        "recording_id": pair.recording_id,
        "status": "ok",
        "fastr_vhdr": str(pair.fastr_vhdr),
        "analyzer_input_vhdr": str(pair.analyzer_input_vhdr),
        "analyzer_output_vhdr": str(pair.analyzer_output_vhdr),
        "input_hashes": {
            "fastr": _recording_hashes(fastr_recording),
            "analyzer_input": _recording_hashes(analyzer_input_recording),
            "analyzer_output": _recording_hashes(analyzer_output_recording),
        },
        "geometry": {
            "fastr_sample_count": int(fastr.data_volts.shape[1]),
            "analyzer_input_sample_count": int(
                analyzer_input.data_volts.shape[1]
            ),
            "analyzer_output_sample_count": int(
                analyzer_output.data_volts.shape[1]
            ),
            "comparison_sample_count": comparison_sample_count,
            "ecg_interior_correlation": ecg_correlation,
        },
        "channel_names": list(fastr.channel_names),
        "eeg_channel_names": [fastr.channel_names[int(index)] for index in eeg_picks],
        "detector": {
            "peak_samples": detection.peak_samples.tolist(),
            "quality": asdict(detection.quality),
        },
        "marker_audit": {
            "analyzer_marker_count": int(analyzer_samples.size),
            "detected_marker_count": int(detection.peak_samples.size),
            "matched_count": marker_audit.matched_count,
            "tolerance_samples": marker_audit.tolerance_samples,
            "median_lag_samples": marker_audit.median_lag_samples,
            "lag_iqr_samples": marker_audit.lag_iqr_samples,
        },
        "correction": {
            "window_seconds": list(config.correction_window_seconds),
            "ecg_to_bcg_delay_seconds": applied_delay,
            "configured_delay_seconds": config.ecg_to_bcg_delay_seconds,
            "artifact_anchor_samples": artifact_anchors.tolist(),
            "detected_event_count": int(detection.peak_samples.size),
            "corrected_event_count": int(correction_peak_samples.size),
            "analyzer_scored_event_count": int(analyzer_metric_samples.size),
            "null_surrogate_count": config.null_surrogate_count,
            "random_seed": config.random_seed,
        },
        "methods": method_rows,
    }


def _read_raw_snapshot(path: Path) -> _RawSnapshot:
    raw = mne.io.read_raw_brainvision(
        path,
        preload=True,
        verbose="ERROR",
    )
    try:
        data = np.asarray(raw.get_data(), dtype=np.float64)
        channel_names = tuple(raw.ch_names)
        channel_types = tuple(raw.get_channel_types())
        sampling_rate_hz = float(raw.info["sfreq"])
    finally:
        raw.close()
    if not np.all(np.isfinite(data)):
        raise BenchmarkInputError(f"recording contains non-finite data: {path}")
    return _RawSnapshot(data, channel_names, channel_types, sampling_rate_hz)


def _channel_index(channel_names: Sequence[str], channel_name: str) -> int:
    try:
        return channel_names.index(channel_name)
    except ValueError as error:
        raise BenchmarkInputError(
            f"configured ECG channel does not exist: {channel_name!r}"
        ) from error


def _validate_pair_geometry(
    fastr: _RawSnapshot,
    analyzer_input: _RawSnapshot,
    analyzer_output: _RawSnapshot,
    ecg_index: int,
    *,
    comparison_band_hz: tuple[float, float],
) -> tuple[float, int]:
    if fastr.channel_names != analyzer_input.channel_names:
        raise BenchmarkInputError("FASTR and Analyzer channel order differs")
    if analyzer_input.channel_names != analyzer_output.channel_names:
        raise BenchmarkInputError(
            "Analyzer input and output channel order differs"
        )
    non_ecg = np.ones(len(fastr.channel_types), dtype=bool)
    non_ecg[ecg_index] = False
    if tuple(np.asarray(fastr.channel_types)[non_ecg]) != tuple(
        np.asarray(analyzer_input.channel_types)[non_ecg]
    ):
        raise BenchmarkInputError("FASTR and Analyzer non-ECG channel types differ")
    if analyzer_input.channel_types != analyzer_output.channel_types:
        raise BenchmarkInputError("Analyzer input and output channel types differ")
    if not np.isclose(
        fastr.sampling_rate_hz,
        analyzer_input.sampling_rate_hz,
        rtol=0.0,
        atol=1e-9,
    ):
        raise BenchmarkInputError("FASTR and Analyzer sampling rates differ")
    if not np.isclose(
        analyzer_input.sampling_rate_hz,
        analyzer_output.sampling_rate_hz,
        rtol=0.0,
        atol=1e-9,
    ):
        raise BenchmarkInputError("Analyzer input and output sampling rates differ")
    analyzer_input_count = analyzer_input.data_volts.shape[1]
    analyzer_output_count = analyzer_output.data_volts.shape[1]
    if analyzer_input_count != analyzer_output_count:
        raise BenchmarkInputError(
            "Analyzer input and output sample geometry differs"
        )
    fastr_count = fastr.data_volts.shape[1]
    if abs(fastr_count - analyzer_input_count) > 1:
        raise BenchmarkInputError("FASTR and Analyzer sample geometry differs")
    comparison_sample_count = min(fastr_count, analyzer_input_count)
    comparison_start = min(
        round(5.0 * fastr.sampling_rate_hz),
        comparison_sample_count // 4,
    )
    fastr_ecg = fastr.data_volts[
        ecg_index,
        comparison_start:comparison_sample_count,
    ]
    analyzer_ecg = analyzer_input.data_volts[
        ecg_index,
        comparison_start:comparison_sample_count,
    ]
    correlation = _band_limited_correlation(
        fastr_ecg,
        analyzer_ecg,
        sampling_rate_hz=fastr.sampling_rate_hz,
        band_hz=comparison_band_hz,
    )
    if correlation < 0.98:
        raise BenchmarkInputError(
            "FASTR and Analyzer ECG vectors have insufficient interior correlation"
        )
    return correlation, comparison_sample_count


def _band_limited_correlation(
    first: np.ndarray,
    second: np.ndarray,
    *,
    sampling_rate_hz: float,
    band_hz: tuple[float, float],
) -> float:
    sos = butter(4, band_hz, btype="bandpass", fs=sampling_rate_hz, output="sos")
    first = sosfiltfilt(sos, first)
    second = sosfiltfilt(sos, second)
    first_centered = first - np.mean(first)
    second_centered = second - np.mean(second)
    denominator = np.linalg.norm(first_centered) * np.linalg.norm(second_centered)
    if denominator == 0.0:
        raise BenchmarkInputError("ECG vectors have no measurable variation")
    return float(np.dot(first_centered, second_centered) / denominator)


def _artifact_anchors(
    peak_samples: np.ndarray,
    sampling_rate_hz: float,
    delay_seconds: float,
    sample_count: int,
) -> np.ndarray:
    anchors = peak_samples + round(delay_seconds * sampling_rate_hz)
    if np.any(anchors < 0) or np.any(anchors >= sample_count):
        raise BenchmarkInputError(
            "ECG-to-BCG delay creates out-of-range artifact anchors"
        )
    return anchors.astype(np.int64, copy=False)


def _complete_event_samples(
    peak_samples: np.ndarray,
    sampling_rate_hz: float,
    delay_seconds: float,
    sample_count: int,
    window_seconds: tuple[float, float],
) -> np.ndarray:
    delay_samples = round(delay_seconds * sampling_rate_hz)
    window_start = round(window_seconds[0] * sampling_rate_hz)
    window_stop = round(window_seconds[1] * sampling_rate_hz)
    anchors = peak_samples + delay_samples
    complete = (
        (anchors + window_start >= 0)
        & (anchors + window_stop <= sample_count)
    )
    return peak_samples[complete]


def _eeg_picks(channel_types: Sequence[str], ecg_index: int) -> np.ndarray:
    picks = np.asarray(
        [
            index
            for index, channel_type in enumerate(channel_types)
            if channel_type == "eeg" and index != ecg_index
        ],
        dtype=np.int64,
    )
    if picks.size == 0:
        raise BenchmarkInputError("recording has no EEG channels after ECG exclusion")
    return picks


def _method_metrics(
    before_data_volts: np.ndarray,
    after_data_volts: np.ndarray,
    eeg_picks: np.ndarray,
    ecg_index: int,
    artifact_anchors: np.ndarray,
    sampling_rate_hz: float,
    window_seconds: tuple[float, float],
    null_values: np.ndarray,
    *,
    corrected_samples: np.ndarray | None,
) -> dict[str, object]:
    before_eeg_uv = before_data_volts[eeg_picks] * 1e6
    after_eeg_uv = after_data_volts[eeg_picks] * 1e6
    before_rms = held_out_cardiac_rms(
        before_eeg_uv,
        artifact_anchors,
        sampling_rate_hz=sampling_rate_hz,
        window_seconds=window_seconds,
    )
    after_rms = held_out_cardiac_rms(
        after_eeg_uv,
        artifact_anchors,
        sampling_rate_hz=sampling_rate_hz,
        window_seconds=window_seconds,
    )
    ratios = cardiac_residual_ratio(
        before_eeg_uv,
        after_eeg_uv,
        artifact_anchors,
        sampling_rate_hz=sampling_rate_hz,
        window_seconds=window_seconds,
    )
    if (
        null_values.ndim != 2
        or null_values.shape[0] == 0
        or null_values.shape[1] != eeg_picks.size
    ):
        raise BenchmarkInputError(
            "null_values must have shape (surrogates, EEG channels)"
        )
    result: dict[str, object] = {
        "before_held_out_rms_uv": before_rms.tolist(),
        "after_held_out_rms_uv": after_rms.tolist(),
        "cardiac_residual_ratio": ratios.tolist(),
        "null_maximum_rms_uv": np.max(null_values, axis=0).tolist(),
        "ecg_maximum_change_uv": float(
            np.max(np.abs(after_data_volts[ecg_index] - before_data_volts[ecg_index]))
            * 1e6
        ),
    }
    if corrected_samples is not None:
        outside = np.ones(before_data_volts.shape[1], dtype=bool)
        outside[corrected_samples] = False
        if not np.any(outside):
            raise BenchmarkInputError(
                "correction windows cover the complete recording"
            )
        result["corrected_sample_count"] = int(corrected_samples.size)
        result["maximum_outside_change_uv"] = float(
            np.max(
                np.abs(
                    after_data_volts[eeg_picks][:, outside]
                    - before_data_volts[eeg_picks][:, outside]
                )
            )
            * 1e6
        )
    return result


def _recording_hashes(recording: BrainVisionRecording) -> dict[str, str]:
    paths = {
        "vhdr": recording.header_path,
        "eeg": recording.data_path,
        "vmrk": recording.marker_path,
    }
    return {name: _sha256(path) for name, path in paths.items()}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    with path.open("x", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2, default=_json_default)
        output_file.write("\n")


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer | np.floating):
        return value.item()
    raise TypeError(f"value is not JSON serializable: {type(value).__name__}")


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    fieldnames = (
        "recording_id",
        "status",
        "fastr_vhdr",
        "analyzer_input_vhdr",
        "analyzer_output_vhdr",
        "error",
        "detector_peak_count",
        "analyzer_marker_count",
        "matched_count",
        "methods_json",
    )
    with path.open("x", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            detector = row.get("detector", {})
            marker_audit = row.get("marker_audit", {})
            writer.writerow(
                {
                    "recording_id": row.get("recording_id", ""),
                    "status": row.get("status", ""),
                    "fastr_vhdr": row.get("fastr_vhdr", ""),
                    "analyzer_input_vhdr": row.get("analyzer_input_vhdr", ""),
                    "analyzer_output_vhdr": row.get("analyzer_output_vhdr", ""),
                    "error": row.get("error", ""),
                    "detector_peak_count": (
                        len(detector.get("peak_samples", []))
                        if isinstance(detector, dict)
                        else ""
                    ),
                    "analyzer_marker_count": (
                        marker_audit.get("analyzer_marker_count", "")
                        if isinstance(marker_audit, dict)
                        else ""
                    ),
                    "matched_count": (
                        marker_audit.get("matched_count", "")
                        if isinstance(marker_audit, dict)
                        else ""
                    ),
                    "methods_json": json.dumps(
                        row.get("methods", {}),
                        default=_json_default,
                    ),
                }
            )


def run_bcg_benchmark(config: BenchmarkConfig) -> BenchmarkSummary:
    pairs = discover_recording_pairs(
        config.fastr_root,
        config.analyzer_input_root,
        config.analyzer_output_root,
    )
    config.output_root.mkdir(parents=True, exist_ok=True)
    report_json = config.output_root / "bcg_benchmark.json"
    report_csv = config.output_root / "bcg_benchmark.csv"
    if report_json.exists() or report_csv.exists():
        raise FileExistsError(
            f"benchmark report already exists in {config.output_root}"
        )

    rows: list[dict[str, object]] = []
    for pair in pairs:
        try:
            rows.append(_benchmark_pair(pair, config))
        except _EXPECTED_RUN_ERRORS as error:
            rows.append(
                {
                    "recording_id": pair.recording_id,
                    "status": "failed",
                    "fastr_vhdr": str(pair.fastr_vhdr),
                    "analyzer_input_vhdr": str(pair.analyzer_input_vhdr),
                    "analyzer_output_vhdr": str(pair.analyzer_output_vhdr),
                    "error": f"{type(error).__name__}: {error}",
                }
            )

    successful_count = sum(row["status"] == "ok" for row in rows)
    failed_count = len(rows) - successful_count
    payload = {
        "fastr_root": str(config.fastr_root),
        "analyzer_input_root": str(config.analyzer_input_root),
        "analyzer_output_root": str(config.analyzer_output_root),
        "output_root": str(config.output_root),
        "configuration": asdict(config),
        "run_count": len(rows),
        "successful_count": successful_count,
        "failed_count": failed_count,
        "runs": rows,
    }
    _write_json(report_json, payload)
    _write_csv(report_csv, rows)
    return BenchmarkSummary(
        report_json=report_json,
        report_csv=report_csv,
        run_count=len(rows),
        successful_count=successful_count,
        failed_count=failed_count,
    )
