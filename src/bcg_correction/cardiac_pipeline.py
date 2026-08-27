"""Pipeline entry point for independent ECG marker derivation."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import mne
import numpy as np

from .bcg_config import DetectionRunConfig
from .brainvision_io import read_brainvision_recording
from .cardiac import CardiacDetection, CardiacInputError, detect_r_peaks
from .cardiac_markers import (
    DetectionSummary,
    validate_fastr_marker_input,
    write_marker_recording,
)


def run_cardiac_detection(config: DetectionRunConfig) -> DetectionSummary:
    """Derive ECG R markers and write a provenance-bearing BrainVision copy."""
    source = read_brainvision_recording(config.input_vhdr)
    validate_fastr_marker_input(source.markers)
    output_vhdr = config.output_vhdr.expanduser().resolve()
    provenance_json = output_vhdr.with_suffix(".cardiac.json")
    _ensure_outputs_are_absent(output_vhdr, provenance_json)

    ecg, sampling_rate_hz = _read_ecg(
        config.input_vhdr,
        channel_name=config.detector.ecg_channel,
    )
    detection = detect_r_peaks(
        ecg,
        sampling_rate_hz,
        config=config.detector,
    )
    write_marker_recording(
        config.input_vhdr,
        output_vhdr,
        peak_samples=detection.peak_samples,
    )
    _write_provenance(
        provenance_json,
        config=config,
        detection=detection,
        sampling_rate_hz=sampling_rate_hz,
    )
    return DetectionSummary(
        output_vhdr=output_vhdr,
        provenance_json=provenance_json,
        marker_count=int(detection.peak_samples.size),
        status=detection.quality.status,
    )


def _read_ecg(
    vhdr_path: Path,
    *,
    channel_name: str,
) -> tuple[np.ndarray, float]:
    raw = mne.io.read_raw_brainvision(
        vhdr_path,
        preload=False,
        verbose="ERROR",
    )
    try:
        try:
            channel_index = raw.ch_names.index(channel_name)
        except ValueError as error:
            raise CardiacInputError(
                f"configured ECG channel does not exist: {channel_name!r}"
            ) from error
        ecg = np.asarray(raw.get_data(picks=[channel_index])[0], dtype=np.float64)
        sampling_rate_hz = float(raw.info["sfreq"])
    finally:
        raw.close()
    return ecg, sampling_rate_hz


def _ensure_outputs_are_absent(
    output_vhdr: Path,
    provenance_json: Path,
) -> None:
    output_paths = (
        output_vhdr,
        output_vhdr.with_suffix(".eeg"),
        output_vhdr.with_suffix(".vmrk"),
        provenance_json,
    )
    existing = tuple(path for path in output_paths if path.exists())
    if existing:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"output already exists: {joined}")


def _write_provenance(
    path: Path,
    *,
    config: DetectionRunConfig,
    detection: CardiacDetection,
    sampling_rate_hz: float,
) -> None:
    payload = {
        "input_vhdr": str(config.input_vhdr),
        "output_vhdr": str(config.output_vhdr),
        "sampling_rate_hz": sampling_rate_hz,
        "detector": asdict(config.detector),
        "peak_samples": detection.peak_samples.tolist(),
        "quality": asdict(detection.quality),
    }
    with path.open("x", encoding="utf-8") as provenance_file:
        json.dump(payload, provenance_file, indent=2)
        provenance_file.write("\n")
