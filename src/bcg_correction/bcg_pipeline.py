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
    correct_bcg,
    rr_gap_spans,
)
from .bcg_config import CorrectionRunConfig
from .brainvision import BrainVisionMarker
from .brainvision_io import read_brainvision_recording, write_brainvision_recording
from .cardiac import CardiacDetection, CardiacInputError, detect_r_peaks
from .cardiac_markers import append_pulse_markers, validate_fastr_marker_input
from .correction_report import (
    compute_correction_profile,
    profile_metrics,
    save_correction_report,
    write_profile,
)
from .metrics import (
    RECORDING_DELAY_WINDOW_SECONDS,
    BcgDelayScan,
    delay_estimation_eeg,
    estimate_ecg_to_bcg_delay,
)


@dataclass(frozen=True, slots=True)
class BcgCorrectionSummary:
    """Output paths and detector quality for one BCG correction."""

    output_vhdr: Path
    provenance_json: Path
    method: str
    marker_count: int
    status: str
    applied_delay_seconds: float


#: The one degradation that does not make a recording uncorrectable.
#:
#: No template is subtracted where there is no detected beat, so an RR gap
#: leaves full-amplitude BCG inside a bounded, identifiable span rather than
#: corrupting anything. Those spans are marked bad in the output for downstream
#: rejection, and their total share is capped by ``maximum_gap_fraction``.
#:
#: Every other degradation is fatal, and ``rr_below_minimum`` is why: an
#: interval shorter than physiology means a spurious detection, which
#: subtracts a template at a time when no beat occurred and so *injects*
#: artifact. One failure mode leaves data uncorrected; the other makes it
#: wrong, and only the first has a bounded downstream remedy.
GAP_DEGRADATION_REASON = "rr_above_maximum"


def _require_usable_detection(
    detection: CardiacDetection,
) -> None:
    reasons = detection.quality.degradation_reasons
    if detection.quality.status == "ok":
        if reasons:
            raise CardiacInputError("inconsistent ECG detection quality")
        return
    if not reasons:
        raise CardiacInputError("degraded ECG detection")
    fatal = tuple(reason for reason in reasons if reason != GAP_DEGRADATION_REASON)
    if fatal:
        detail = ", ".join(fatal)
        raise CardiacInputError(f"degraded ECG detection: {detail}")


def _require_tolerable_gaps(
    gap_spans: tuple[tuple[int, int], ...],
    *,
    sample_count: int,
    maximum_gap_fraction: float,
) -> float:
    """Return the share of the recording inside RR gaps, refusing too much.

    A recording whose gaps dominate is not one with a few uncorrected spans;
    it is one whose detected beats cannot be trusted either, so it is refused
    rather than quietly half-corrected.
    """
    if sample_count <= 0:
        raise CardiacInputError("recording has no samples")
    covered = sum(stop - start for start, stop in gap_spans)
    fraction = covered / sample_count
    if fraction > maximum_gap_fraction:
        raise CardiacInputError(
            f"RR gaps cover {fraction:.2%} of the recording across "
            f"{len(gap_spans)} gap(s), above the "
            f"{maximum_gap_fraction:.2%} maximum"
        )
    return fraction


def _output_paths(output_vhdr: Path) -> dict[str, Path]:
    stem = output_vhdr.with_suffix("")
    return {
        "vhdr": output_vhdr,
        "eeg": output_vhdr.with_suffix(".eeg"),
        "vmrk": output_vhdr.with_suffix(".vmrk"),
        "report": Path(f"{stem}_correction_report.png"),
        "profile": Path(f"{stem}_profile.npz"),
        "json": output_vhdr.with_suffix(".bcg.json"),
    }


def _channel_index(names: tuple[str, ...], channel_name: str) -> int:
    try:
        return names.index(channel_name)
    except ValueError as error:
        raise CardiacInputError(
            f"configured ECG channel does not exist: {channel_name!r}"
        ) from error


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
    gap_fraction: float,
    delay_scan: BcgDelayScan,
    residual_quality: dict,
    output_paths: dict[str, Path],
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
        "rr_gap_fraction": gap_fraction,
        "maximum_gap_fraction": config.maximum_gap_fraction,
        "residual_qc": residual_quality,
        "evaluation": asdict(config.evaluation),
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
        json.dump(payload, provenance_file, indent=2, allow_nan=False)
        provenance_file.write("\n")


def run_bcg_correction(config: CorrectionRunConfig) -> BcgCorrectionSummary:
    """Detect independent R peaks and subtract BCG from EEG channels only."""
    source = read_brainvision_recording(config.input_vhdr)
    validate_fastr_marker_input(source.markers)
    output_vhdr = config.output_vhdr.expanduser().resolve()
    output_paths = _output_paths(output_vhdr)
    _ensure_outputs_are_absent(output_paths)

    raw = mne.io.read_raw_brainvision(config.input_vhdr, preload=True, verbose="ERROR")
    try:
        names = tuple(raw.ch_names)
        sampling_rate_hz = float(raw.info["sfreq"])
        data = np.asarray(raw.get_data(), dtype=np.float64)
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
    rr_gaps = rr_gap_spans(
        detection.peak_samples,
        sampling_rate_hz,
        config.detector.maximum_rr_seconds,
    )
    gap_fraction = _require_tolerable_gaps(
        rr_gaps,
        sample_count=data.shape[1],
        maximum_gap_fraction=config.maximum_gap_fraction,
    )
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
    profile = compute_correction_profile(
        data,
        correction.data_volts,
        names,
        ecg_channel_index=ecg_index,
        peak_samples=detection.peak_samples,
        sampling_rate_hz=sampling_rate_hz,
        delay_seconds=delay_scan.best_delay_seconds,
        window_seconds=config.window_seconds,
        gap_fraction=gap_fraction,
        method=config.method,
        label=config.input_vhdr.stem,
        subject=output_vhdr.parent.name,
        evaluation=config.evaluation,
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
    _write_provenance(
        output_paths["json"],
        config=config,
        detection=detection,
        correction=correction,
        sampling_rate_hz=sampling_rate_hz,
        gap_spans=rr_gaps,
        gap_fraction=gap_fraction,
        delay_scan=delay_scan,
        residual_quality={
            "status": "descriptive_only_not_an_acceptance_gate",
            "preservation_status": "not_measured",
            "profile": str(output_paths["profile"]) if profile is not None else None,
            "metrics": profile_metrics(profile) if profile is not None else None,
        },
        output_paths=output_paths,
    )
    # Called after provenance on purpose: the page is a diagnostic, and a
    # plotting failure must not cost the record of what was written.
    if profile is not None:
        write_profile(profile, output_paths["profile"])
        save_correction_report(
            profile,
            title=(
                f"{output_vhdr.stem}  \u2014  {config.method.upper()} correction report"
            ),
            output=output_paths["report"],
        )
    return BcgCorrectionSummary(
        output_vhdr=output_paths["vhdr"],
        provenance_json=output_paths["json"],
        method=config.method,
        marker_count=int(detection.peak_samples.size),
        status=detection.quality.status,
        applied_delay_seconds=delay_scan.best_delay_seconds,
    )
