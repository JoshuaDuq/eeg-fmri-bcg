"""Raw vs corrected-arm overlays and the per-recording metrics row."""

from __future__ import annotations

from pathlib import Path

import mne
import numpy as np
from scipy.signal import welch

from .arms import BCGNET, CLEAN_ARMS, COMPARATOR_ARMS
from .pairs import RecordingSet
from .qc import (
    alpha_peak_height,
    load_shared_detector_provenance,
    median_locked_ratio,
    method_qc_flags,
    removal_profile,
)

RAW_LABEL = "Raw"

_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
}

_PROFILE_METRICS = (
    "specificity",
    "alpha_collateral_fraction",
    "locked_removed_uv",
    "collateral_uv",
)
_FLAG_COLUMNS = (
    "bcgnet_adds_power",
    "bcgnet_locked_worse_than_raw",
    "alpha_peak_collapsed",
    "prefer_comparator",
)


def _metric_columns() -> tuple[str, ...]:
    columns = ["bids_id", "stem", "label", "run"]
    columns.extend(f"has_{arm.key}" for arm in CLEAN_ARMS)
    columns.append("rms_raw")
    for band in _BANDS:
        columns.append(f"{band}_raw")
        for arm in CLEAN_ARMS:
            columns.extend((f"{band}_{arm.key}", f"{band}_{arm.key}_ratio"))
    columns.append("alpha_peak_raw")
    for arm in CLEAN_ARMS:
        columns.extend((f"rms_{arm.key}", f"alpha_peak_{arm.key}"))
    for arm in CLEAN_ARMS:
        columns.append(f"locked_{arm.key}_ratio")
        columns.extend(f"{name}_{arm.key}" for name in _PROFILE_METRICS)
    columns.extend(_FLAG_COLUMNS)
    return tuple(columns)


METRIC_COLUMNS = _metric_columns()


def load_fastr(path: Path) -> mne.io.BaseRaw:
    return mne.io.read_raw_brainvision(path, preload=True, verbose="ERROR")


def _eeg_indices(raw: mne.io.BaseRaw) -> np.ndarray:
    names = raw.ch_names
    if "ECG" in names:
        return np.array(
            [index for index, name in enumerate(names) if name != "ECG"]
        )
    return np.arange(len(names))


def mean_eeg_psd(
    raw: mne.io.BaseRaw, *, max_hz: float
) -> tuple[np.ndarray, np.ndarray]:
    data = raw.get_data(picks=_eeg_indices(raw)) * 1e6
    fs = float(raw.info["sfreq"])
    nperseg = min(int(fs * 3), data.shape[1])
    freqs, pxx = welch(data, fs=fs, nperseg=nperseg, axis=1)
    keep = freqs <= max_hz
    return freqs[keep], np.mean(pxx[:, keep], axis=0)


def band_power(freqs: np.ndarray, pxx: np.ndarray, low: float, high: float) -> float:
    mask = (freqs >= low) & (freqs <= high)
    return float(np.sum(pxx[mask]))


def detector_provenance(recording: RecordingSet):
    return load_shared_detector_provenance(
        {
            arm.key: recording.cleaned_vhdr[arm.key]
            for arm in COMPARATOR_ARMS
            if arm.key in recording.cleaned_vhdr
        }
    )


def metrics_row(
    recording: RecordingSet,
    traces: dict[str, mne.io.BaseRaw],
    *,
    max_hz: float,
) -> dict[str, object]:
    row: dict[str, object] = {
        "bids_id": recording.bids_id,
        "stem": recording.stem,
        "label": recording.label,
        "run": recording.run,
    }
    for arm in CLEAN_ARMS:
        row[f"has_{arm.key}"] = arm.label in traces

    psds = {
        name: mean_eeg_psd(raw, max_hz=max_hz) for name, raw in traces.items()
    }
    raw_f, raw_p = psds[RAW_LABEL]
    row["rms_raw"] = float(
        np.sqrt(np.mean(np.square(traces[RAW_LABEL].get_data() * 1e6)))
    )

    remaining: dict[str, float | None] = {}
    for band, (low, high) in _BANDS.items():
        raw_band = band_power(raw_f, raw_p, low, high)
        row[f"{band}_raw"] = raw_band
        for arm in CLEAN_ARMS:
            value = None
            ratio = None
            if arm.label in psds:
                value = band_power(*psds[arm.label], low, high)
                ratio = value / raw_band if raw_band else None
            row[f"{band}_{arm.key}"] = value
            row[f"{band}_{arm.key}_ratio"] = ratio
            if arm is BCGNET:
                remaining[band] = ratio

    alpha_raw = alpha_peak_height(raw_f, raw_p)
    row["alpha_peak_raw"] = alpha_raw
    alpha_net = None
    for arm in CLEAN_ARMS:
        rms = None
        peak = None
        if arm.label in traces:
            rms = float(
                np.sqrt(np.mean(np.square(traces[arm.label].get_data() * 1e6)))
            )
            peak = alpha_peak_height(*psds[arm.label])
        row[f"rms_{arm.key}"] = rms
        row[f"alpha_peak_{arm.key}"] = peak
        if arm is BCGNET:
            alpha_net = peak

    provenance = detector_provenance(recording)
    for arm in CLEAN_ARMS:
        ratio = None
        if provenance is not None and arm.label in traces:
            ratio = median_locked_ratio(
                traces[RAW_LABEL],
                traces[arm.label],
                peak_samples=provenance.peak_samples,
                delay_seconds=provenance.delay_seconds,
                window_seconds=provenance.window_seconds,
            )
        row[f"locked_{arm.key}_ratio"] = ratio
        profile: dict[str, float | None] = {
            "specificity": None,
            "alpha_collateral_fraction": None,
            "locked_removed_uv": None,
            "collateral_uv": None,
        }
        if provenance is not None and arm.label in traces:
            profile = removal_profile(
                traces[RAW_LABEL],
                traces[arm.label],
                peak_samples=provenance.peak_samples,
                delay_seconds=provenance.delay_seconds,
                window_seconds=provenance.window_seconds,
            )
        for name, value in profile.items():
            row[f"{name}_{arm.key}"] = value

    row.update(
        method_qc_flags(
            remaining_ratios=remaining,
            locked_ratio=row[f"locked_{BCGNET.key}_ratio"],
            alpha_peak_raw=alpha_raw,
            alpha_peak_bcgnet=alpha_net,
        )
    )
    if tuple(row) != METRIC_COLUMNS:
        raise RuntimeError("comparison metric row does not match its declared schema")
    return row
