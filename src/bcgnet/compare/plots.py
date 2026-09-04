"""Rectangular, symmetric exports from the same profiles used by the figures."""

from pathlib import Path

import mne
import numpy as np
from scipy.signal import welch

from bcg_correction.correction_report import compute_correction_profile, profile_metrics
from bcg_correction.evaluation import EvaluationSettings, band_integral

from .arms import CLEAN_ARMS, COMPARATOR_ARMS
from .qc import alpha_peak_height, load_shared_detector_provenance

RAW_LABEL = "Raw"
_BANDS = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13)}
_VARIANTS = ("as_written", "ecg_regressed")
_PROFILE_SCALARS = ("locked_removal_fraction", "variable_removal_alpha_ratio")


def metric_columns(evaluation: EvaluationSettings):
    columns = ["bids_id", "stem", "label", "run", "evaluation_scope"]
    columns.extend(f"has_{arm.key}" for arm in CLEAN_ARMS)
    columns.append("rms_raw")
    for band in _BANDS:
        columns.append(f"{band}_raw")
        for arm in CLEAN_ARMS:
            columns.extend((f"{band}_{arm.key}", f"{band}_{arm.key}_ratio"))
    columns.append("alpha_peak_raw")
    for arm in CLEAN_ARMS:
        columns.extend(
            f"{name}_{arm.key}"
            for name in (
                "rms",
                "alpha_peak",
                "evaluation_status",
                "preservation_status",
                "beats",
                "gap_fraction",
                *_PROFILE_SCALARS,
            )
        )
        for count in evaluation.block_counts:
            columns.append(f"local_{count}_minimum_beats_{arm.key}")
            for variant in _VARIANTS:
                columns.extend(
                    f"local_{count}_{variant}_{name}_{arm.key}"
                    for name in ("before_uv", "after_uv", "ratio")
                )
    return tuple(columns)


def load_fastr(path: Path):
    return mne.io.read_raw_brainvision(path, preload=True, verbose="ERROR")


def _eeg_indices(raw):
    return [
        index
        for index in mne.pick_types(raw.info, eeg=True, exclude=[])
        if raw.ch_names[index] != "ECG"
    ]


def mean_eeg_psd(raw, *, max_hz):
    data = raw.get_data(picks=_eeg_indices(raw)) * 1e6
    fs = float(raw.info["sfreq"])
    frequency, power = welch(
        data, fs=fs, nperseg=min(int(fs * 3), data.shape[1]), axis=-1
    )
    keep = frequency <= max_hz
    return frequency[keep], power[:, keep].mean(axis=0)


def band_power(frequency, power, low, high):
    return float(band_integral(frequency, power, low, high))


def detector_provenance(recording):
    return load_shared_detector_provenance(
        {
            arm.key: recording.cleaned_vhdr[arm.key]
            for arm in COMPARATOR_ARMS
            if arm.key in recording.cleaned_vhdr
        }
    )


def measure_recording(recording, traces, evaluation):
    provenance = detector_provenance(recording)
    if provenance is None:
        return {}
    raw = traces[RAW_LABEL]
    if "ECG" not in raw.ch_names:
        raise ValueError("comparison requires the original ECG channel")
    expected_eeg = [i for i in range(len(raw.ch_names)) if raw.ch_names[i] != "ECG"]
    if list(_eeg_indices(raw)) != expected_eeg:
        raise ValueError("comparison inputs must contain only EEG and the ECG channel")
    profiles = {}
    for arm in CLEAN_ARMS:
        if arm.label not in traces:
            continue
        cleaned = traces[arm.label]
        if (
            cleaned.ch_names != raw.ch_names
            or cleaned.n_times != raw.n_times
            or cleaned.info["sfreq"] != raw.info["sfreq"]
        ):
            raise ValueError(f"unaligned comparison input: {arm.label}")
        profile = compute_correction_profile(
            raw.get_data(),
            cleaned.get_data(),
            tuple(raw.ch_names),
            ecg_channel_index=raw.ch_names.index("ECG"),
            peak_samples=provenance.peak_samples,
            sampling_rate_hz=float(raw.info["sfreq"]),
            delay_seconds=provenance.delay_seconds,
            window_seconds=provenance.window_seconds,
            gap_fraction=provenance.gap_fraction,
            method=arm.key,
            label=recording.stem,
            subject=recording.bids_id,
            evaluation=evaluation,
        )
        if profile is not None:
            profiles[arm.key] = profile
    return profiles


def _finite(value):
    return float(value) if value is not None and np.isfinite(value) else None


def metrics_row(recording, traces, profiles, *, max_hz, evaluation):
    row = dict.fromkeys(metric_columns(evaluation))
    row.update(
        bids_id=recording.bids_id,
        stem=recording.stem,
        label=recording.label,
        run=recording.run,
        evaluation_scope="saved_outputs_descriptive_not_independent_validation",
    )
    psds = {name: mean_eeg_psd(raw, max_hz=max_hz) for name, raw in traces.items()}
    raw_frequency, raw_power = psds[RAW_LABEL]
    raw = traces[RAW_LABEL]
    row["rms_raw"] = float(
        np.sqrt(np.mean(raw.get_data(picks=_eeg_indices(raw)) ** 2)) * 1e6
    )
    row["alpha_peak_raw"] = alpha_peak_height(raw_frequency, raw_power)
    for band, (low, high) in _BANDS.items():
        original = band_power(raw_frequency, raw_power, low, high)
        row[f"{band}_raw"] = _finite(original)
        for arm in CLEAN_ARMS:
            if arm.label in psds:
                value = band_power(*psds[arm.label], low, high)
                row[f"{band}_{arm.key}"] = _finite(value)
                row[f"{band}_{arm.key}_ratio"] = (
                    _finite(value / original) if original > 0 else None
                )
    for arm in CLEAN_ARMS:
        present = arm.label in traces
        row[f"has_{arm.key}"] = present
        row[f"evaluation_status_{arm.key}"] = (
            "not_available" if present else "missing_output"
        )
        row[f"preservation_status_{arm.key}"] = "not_measured"
        if present:
            data = traces[arm.label].get_data(picks=_eeg_indices(traces[arm.label]))
            row[f"rms_{arm.key}"] = float(np.sqrt(np.mean(data**2)) * 1e6)
            row[f"alpha_peak_{arm.key}"] = alpha_peak_height(*psds[arm.label])
        if arm.key not in profiles:
            continue
        profile = profiles[arm.key]
        if (
            tuple(profile.block_counts) != evaluation.block_counts
            or profile.minimum_beats_per_block != evaluation.minimum_beats_per_block
        ):
            raise ValueError("profile evaluation settings differ from export settings")
        available = np.isfinite(profile.local_ratio)
        row[f"evaluation_status_{arm.key}"] = (
            "available"
            if available.all()
            else "partial"
            if available.any()
            else "insufficient_beats_or_zero_reference"
        )
        for name, value in profile_metrics(profile).items():
            row[f"{name}_{arm.key}"] = value
    return row
