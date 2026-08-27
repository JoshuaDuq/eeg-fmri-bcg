"""Production BCG correction from independent ECG R markers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import mne
import numpy as np

from .bcg import (
    BcgCorrectionConfig,
    BcgCorrectionResult,
    BcgInputError,
    correct_bcg,
    rr_gap_spans,
)
from .bcg_config import CorrectionRunConfig
from .brainvision import BrainVisionMarker
from .brainvision_io import read_brainvision_recording, write_brainvision_recording
from .cardiac import CardiacDetection, CardiacInputError, detect_r_peaks
from .cardiac_markers import append_pulse_markers, validate_fastr_marker_input
from .metrics import (
    RECORDING_DELAY_WINDOW_SECONDS,
    BcgDelayScan,
    cardiac_locked_rms,
    delay_estimation_eeg,
    estimate_ecg_to_bcg_delay,
)
from .psd import PSD_FFT_SAMPLES, PSD_MAX_FREQUENCY_HZ, save_psd_plot


@dataclass(frozen=True, slots=True)
class BcgCorrectionSummary:
    """Output paths and detector quality for one BCG correction."""

    output_vhdr: Path
    provenance_json: Path
    psd_before: Path
    psd_after: Path
    method: str
    marker_count: int
    status: str
    applied_delay_seconds: float


@dataclass(frozen=True, slots=True)
class BcgResidualQuality:
    """Recording-level heartbeat-locked energy before and after correction."""

    before_median_rms_uv: float
    after_median_rms_uv: float
    ratio: float
    maximum_allowed_ratio: float


def _require_usable_detection(
    detection: CardiacDetection,
) -> None:
    reasons = detection.quality.degradation_reasons
    if detection.quality.status == "ok":
        if reasons:
            raise CardiacInputError("inconsistent ECG detection quality")
        return
    if reasons:
        detail = ", ".join(reasons)
        raise CardiacInputError(f"degraded ECG detection: {detail}")
    raise CardiacInputError("degraded ECG detection")


def run_bcg_correction(config: CorrectionRunConfig) -> BcgCorrectionSummary:
    """Detect independent R peaks and subtract BCG from EEG channels only."""
    source = read_brainvision_recording(config.input_vhdr)
    validate_fastr_marker_input(source.markers)
    output_vhdr = config.output_vhdr.expanduser().resolve()
    output_paths = _output_paths(output_vhdr)
    _ensure_outputs_are_absent(output_paths)

    raw = mne.io.read_raw_brainvision(
        config.input_vhdr, preload=True, verbose="ERROR"
    )
    try:
        names = tuple(raw.ch_names)
        sampling_rate_hz = float(raw.info["sfreq"])
        data = np.asarray(raw.get_data(), dtype=np.float64)
        before_raw = raw.copy()
    finally:
        raw.close()

    ecg_index = _channel_index(names, config.detector.ecg_channel)
    eeg_picks = np.array(
        [index for index in range(len(names)) if index != ecg_index],
        dtype=np.int64,
    )
    detection = detect_r_peaks(
        data[ecg_index],
        sampling_rate_hz,
        config=config.detector,
    )
    _require_usable_detection(detection)
    delay_scan = estimate_ecg_to_bcg_delay(
        delay_estimation_eeg(
            data,
            names,
            ecg_channel_index=ecg_index,
        ),
        detection.peak_samples,
        sampling_rate_hz=sampling_rate_hz,
    )
    correction = correct_bcg(
        data,
        detection.peak_samples,
        sampling_rate_hz,
        channel_names=names,
        eeg_picks=eeg_picks,
        ecg_channel_index=ecg_index,
        config=BcgCorrectionConfig(
            method=config.method,
            window_seconds=config.window_seconds,
            ecg_to_bcg_delay_seconds=delay_scan.best_delay_seconds,
            aas_neighbor_count=config.aas_neighbor_count,
            pca_obs_components=config.pca_obs_components,
        ),
    )
    residual_quality = _measure_residual_quality(
        data,
        correction.data_volts,
        names,
        ecg_index=ecg_index,
        peak_samples=detection.peak_samples,
        sampling_rate_hz=sampling_rate_hz,
        delay_seconds=delay_scan.best_delay_seconds,
        window_seconds=config.window_seconds,
        maximum_ratio=config.maximum_residual_ratio,
    )
    psd_tmin, psd_tmax, psd_n_fft = _bcg_psd_interval(
        correction.corrected_samples,
        sampling_rate_hz=sampling_rate_hz,
        sample_count=data.shape[1],
    )
    rr_gaps = rr_gap_spans(
        detection.peak_samples,
        sampling_rate_hz,
        config.detector.maximum_rr_seconds,
    )
    markers = append_pulse_markers(
        source.markers,
        detection.peak_samples,
        sample_count=data.shape[1],
    ) + _bad_bcg_markers(rr_gaps, sample_count=data.shape[1])
    write_brainvision_recording(
        data=correction.data_volts,
        sampling_rate=sampling_rate_hz,
        channel_names=names,
        output_vhdr=output_paths["vhdr"],
        markers=markers,
    )
    _save_bcg_psd_plots(
        before_raw,
        output_paths["vhdr"],
        output_paths,
        sampling_rate_hz=sampling_rate_hz,
        psd_tmin=psd_tmin,
        psd_tmax=psd_tmax,
        psd_n_fft=psd_n_fft,
    )
    _write_provenance(
        output_paths["json"],
        config=config,
        detection=detection,
        correction=correction,
        sampling_rate_hz=sampling_rate_hz,
        gap_spans=rr_gaps,
        delay_scan=delay_scan,
        residual_quality=residual_quality,
        output_paths=output_paths,
        psd_tmin=psd_tmin,
        psd_tmax=psd_tmax,
    )
    return BcgCorrectionSummary(
        output_vhdr=output_paths["vhdr"],
        provenance_json=output_paths["json"],
        psd_before=output_paths["psd_before"],
        psd_after=output_paths["psd_after"],
        method=config.method,
        marker_count=int(detection.peak_samples.size),
        status=detection.quality.status,
        applied_delay_seconds=delay_scan.best_delay_seconds,
    )


def _output_paths(output_vhdr: Path) -> dict[str, Path]:
    stem = output_vhdr.with_suffix("")
    return {
        "vhdr": output_vhdr,
        "eeg": output_vhdr.with_suffix(".eeg"),
        "vmrk": output_vhdr.with_suffix(".vmrk"),
        "json": output_vhdr.with_suffix(".bcg.json"),
        "psd_before": stem.with_name(f"{stem.name}_psd_before.png"),
        "psd_after": stem.with_name(f"{stem.name}_psd_after.png"),
    }


def _bcg_psd_interval(
    corrected_samples: np.ndarray,
    *,
    sampling_rate_hz: float,
    sample_count: int,
) -> tuple[float, float, int]:
    values = np.asarray(corrected_samples)
    if values.ndim != 1 or values.size < 2:
        raise BcgInputError("BCG PSD requires at least two corrected samples")
    if not np.issubdtype(values.dtype, np.integer):
        raise BcgInputError("BCG corrected samples must be integer positions")
    values = values.astype(np.int64, copy=False)
    if np.any(values < 0) or np.any(values >= sample_count):
        raise BcgInputError("BCG corrected samples exceed the recording")
    if np.any(np.diff(values) <= 0):
        raise BcgInputError("BCG corrected samples must be strictly increasing")
    runs = np.split(values, np.flatnonzero(np.diff(values) > 1) + 1)
    longest = max(runs, key=lambda run: run.size)
    span_samples = int(longest.size)
    n_fft = min(PSD_FFT_SAMPLES, span_samples)
    if n_fft < 2:
        raise BcgInputError("BCG PSD requires two samples per corrected window")
    return (
        float(longest[0]) / sampling_rate_hz,
        float(longest[-1] + 1) / sampling_rate_hz,
        n_fft,
    )


def _prepare_bcg_psd_raw(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    prepared = raw.copy()
    keep = prepared.annotations.description != "Bad Interval/Bad_Gradient"
    prepared.set_annotations(prepared.annotations[keep])
    return prepared


def _save_bcg_psd_plots(
    before_raw: mne.io.BaseRaw,
    output_vhdr: Path,
    output_paths: dict[str, Path],
    *,
    sampling_rate_hz: float,
    psd_tmin: float,
    psd_tmax: float,
    psd_n_fft: int,
) -> None:
    corrected_raw = mne.io.read_raw_brainvision(
        output_vhdr,
        preload=False,
        verbose="ERROR",
    )
    before_psd_raw = _prepare_bcg_psd_raw(before_raw)
    after_psd_raw = _prepare_bcg_psd_raw(corrected_raw)
    fmax = min(PSD_MAX_FREQUENCY_HZ, sampling_rate_hz / 2.0)
    try:
        save_psd_plot(
            before_psd_raw,
            output_paths["psd_before"],
            title="Before BCG correction",
            fmax=fmax,
            tmin=psd_tmin,
            tmax=psd_tmax,
            n_fft=psd_n_fft,
        )
        save_psd_plot(
            after_psd_raw,
            output_paths["psd_after"],
            title="After BCG correction",
            fmax=fmax,
            tmin=psd_tmin,
            tmax=psd_tmax,
            n_fft=psd_n_fft,
        )
    finally:
        after_psd_raw.close()
        before_psd_raw.close()
        corrected_raw.close()
        before_raw.close()


def _channel_index(names: tuple[str, ...], channel_name: str) -> int:
    try:
        return names.index(channel_name)
    except ValueError as error:
        raise CardiacInputError(
            f"configured ECG channel does not exist: {channel_name!r}"
        ) from error


def _measure_residual_quality(
    before_data_volts: np.ndarray,
    after_data_volts: np.ndarray,
    channel_names: tuple[str, ...],
    *,
    ecg_index: int,
    peak_samples: np.ndarray,
    sampling_rate_hz: float,
    delay_seconds: float,
    window_seconds: tuple[float, float],
    maximum_ratio: float,
) -> BcgResidualQuality:
    locked_window = (
        delay_seconds + window_seconds[0],
        delay_seconds + window_seconds[1],
    )
    before_eeg_uv = delay_estimation_eeg(
        before_data_volts,
        channel_names,
        ecg_channel_index=ecg_index,
    ) * 1e6
    after_eeg_uv = delay_estimation_eeg(
        after_data_volts,
        channel_names,
        ecg_channel_index=ecg_index,
    ) * 1e6
    before_rms = cardiac_locked_rms(
        before_eeg_uv,
        peak_samples,
        sampling_rate_hz=sampling_rate_hz,
        window_seconds=locked_window,
    )
    after_rms = cardiac_locked_rms(
        after_eeg_uv,
        peak_samples,
        sampling_rate_hz=sampling_rate_hz,
        window_seconds=locked_window,
    )
    before_median = float(np.median(before_rms))
    if before_median == 0.0:
        raise BcgInputError("BCG residual ratio has a zero before-correction RMS")
    after_median = float(np.median(after_rms))
    ratio = after_median / before_median
    if ratio > maximum_ratio:
        raise BcgInputError(
            f"BCG residual ratio {ratio:.3f} exceeds maximum "
            f"{maximum_ratio:.3f}"
        )
    return BcgResidualQuality(
        before_median_rms_uv=before_median,
        after_median_rms_uv=after_median,
        ratio=ratio,
        maximum_allowed_ratio=maximum_ratio,
    )


def _ensure_outputs_are_absent(output_paths: dict[str, Path]) -> None:
    existing = tuple(path for path in output_paths.values() if path.exists())
    if existing:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"output already exists: {joined}")


def _bad_bcg_markers(
    spans: tuple[tuple[int, int], ...],
    *,
    sample_count: int,
) -> tuple[BrainVisionMarker, ...]:
    markers = []
    for start, stop in spans:
        clipped_start = max(int(start), 0)
        clipped_stop = min(int(stop), sample_count)
        if clipped_stop <= clipped_start:
            continue
        markers.append(
            BrainVisionMarker(
                marker_type="Bad Interval",
                description="Bad_BCG",
                position=clipped_start + 1,
                size=clipped_stop - clipped_start,
                channel=0,
            )
        )
    return tuple(markers)


def _write_provenance(
    path: Path,
    *,
    config: CorrectionRunConfig,
    detection: CardiacDetection,
    correction: BcgCorrectionResult,
    sampling_rate_hz: float,
    gap_spans: tuple[tuple[int, int], ...],
    delay_scan: BcgDelayScan,
    residual_quality: BcgResidualQuality,
    output_paths: dict[str, Path],
    psd_tmin: float,
    psd_tmax: float,
) -> None:
    payload = {
        "input_vhdr": str(config.input_vhdr),
        "output_vhdr": str(config.output_vhdr),
        "sampling_rate_hz": sampling_rate_hz,
        "method": config.method,
        "window_seconds": list(config.window_seconds),
        "ecg_to_bcg_delay_seconds": delay_scan.best_delay_seconds,
        "aas_neighbor_count": config.aas_neighbor_count,
        "pca_obs_components": config.pca_obs_components,
        "detector": asdict(config.detector),
        "peak_samples": detection.peak_samples.tolist(),
        "quality": asdict(detection.quality),
        "corrected_sample_count": int(correction.corrected_samples.size),
        "rr_gap_spans": [list(span) for span in gap_spans],
        "residual_qc": asdict(residual_quality),
        "psd_before": str(output_paths["psd_before"]),
        "psd_after": str(output_paths["psd_after"]),
        "psd_interval_seconds": {
            "start": psd_tmin,
            "end": psd_tmax,
        },
        "delay_estimation": {
            "configured_delay_seconds": config.ecg_to_bcg_delay_seconds,
            "best_delay_seconds": delay_scan.best_delay_seconds,
            "delays_seconds": list(delay_scan.delays_seconds),
            "window_seconds": list(RECORDING_DELAY_WINDOW_SECONDS),
            "channels": "ecg_regressed_posterior",
            "median_locked_rms": list(delay_scan.median_locked_rms),
        },
    }
    with path.open("x", encoding="utf-8") as provenance_file:
        json.dump(payload, provenance_file, indent=2)
        provenance_file.write("\n")
