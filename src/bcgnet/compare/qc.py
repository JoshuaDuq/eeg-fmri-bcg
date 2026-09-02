"""QC numbers per arm: leftover bands, locked residual, alpha peak."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import numpy.typing as npt
from scipy.signal import welch

from bcg_correction.metrics import cardiac_locked_rms, delay_estimation_eeg
from bcg_correction.provenance import (
    CorrectionProvenance,
    load_correction_provenance,
)

ALPHA_PEAK_BAND = (8.0, 13.0)
ALPHA_PEAK_COLLAPSE = 0.5
#: Below this, a locked average is too noisy to split into locked and not.
MINIMUM_PROFILE_BEATS = 8


def remaining_ratio(after: float, before: float) -> float | None:
    if before == 0:
        return None
    return float(after / before)


def alpha_peak_height(
    freqs: npt.NDArray[np.floating],
    pxx: npt.NDArray[np.floating],
    *,
    low: float = ALPHA_PEAK_BAND[0],
    high: float = ALPHA_PEAK_BAND[1],
) -> float | None:
    mask = (freqs >= low) & (freqs <= high)
    if not np.any(mask):
        return None
    return float(np.max(pxx[mask]))


def method_qc_flags(
    *,
    remaining_ratios: dict[str, float | None],
    locked_ratio: float | None,
    alpha_peak_raw: float | None,
    alpha_peak_bcgnet: float | None,
) -> dict[str, bool]:
    adds_power = any(
        value is not None and value > 1.0 for value in remaining_ratios.values()
    )
    locked_worse = locked_ratio is not None and locked_ratio > 1.0
    collapsed = (
        alpha_peak_raw is not None
        and alpha_peak_raw > 0
        and alpha_peak_bcgnet is not None
        and (alpha_peak_bcgnet / alpha_peak_raw) < ALPHA_PEAK_COLLAPSE
    )
    return {
        "bcgnet_adds_power": bool(adds_power),
        "bcgnet_locked_worse_than_raw": bool(locked_worse),
        "alpha_peak_collapsed": bool(collapsed),
        # Alpha-band peak height often tracks BCG harmonics, so collapse
        # is reported but does not by itself send you to a comparator.
        "prefer_comparator": bool(adds_power or locked_worse),
    }


def _alpha_power(data: np.ndarray, sampling_rate: float, nperseg: int) -> float:
    freqs, power = welch(data, fs=sampling_rate, nperseg=nperseg, axis=-1)
    inside = (freqs >= ALPHA_PEAK_BAND[0]) & (freqs <= ALPHA_PEAK_BAND[1])
    return float(np.median(power[:, inside].sum(axis=1)))


def removal_profile(
    raw,
    cleaned,
    *,
    peak_samples: npt.NDArray[np.integer],
    delay_seconds: float,
    window_seconds: tuple[float, float],
) -> dict[str, float | None]:
    empty: dict[str, float | None] = {
        "specificity": None,
        "alpha_collateral_fraction": None,
        "locked_removed_uv": None,
        "collateral_uv": None,
    }
    if "ECG" not in raw.ch_names:
        return empty
    ecg_index = raw.ch_names.index("ECG")
    sampling_rate = float(raw.info["sfreq"])
    window = (
        delay_seconds + window_seconds[0],
        delay_seconds + window_seconds[1],
    )
    start = round(window[0] * sampling_rate)
    span = round(window[1] * sampling_rate) - start
    before = delay_estimation_eeg(
        raw.get_data(), raw.ch_names, ecg_channel_index=ecg_index
    ) * 1e6
    after = delay_estimation_eeg(
        cleaned.get_data(), cleaned.ch_names, ecg_channel_index=ecg_index
    ) * 1e6
    if before.shape != after.shape or span < 2:
        return empty
    before = before - np.median(before, axis=1, keepdims=True)
    after = after - np.median(after, axis=1, keepdims=True)
    starts = np.asarray(peak_samples, dtype=np.int64) + start
    keep = (starts >= 0) & (starts + span <= before.shape[1])
    starts = starts[keep]
    if starts.size < MINIMUM_PROFILE_BEATS:
        return empty
    index = starts[:, None] + np.arange(span)[None, :]
    removed = (before - after)[:, index]
    locked = removed.mean(axis=1)
    nonlocked = removed - locked[:, None, :]
    total_rms = np.sqrt(np.mean(removed**2, axis=(1, 2)))
    locked_rms = np.sqrt(np.mean(locked**2, axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        specificity = float(
            np.median(locked_rms / np.where(total_rms > 0, total_rms, np.nan))
        )
    channels = removed.shape[0]
    nperseg = int(min(span, 1024))
    raw_alpha = _alpha_power(
        before[:, index].reshape(channels, -1), sampling_rate, nperseg
    )
    collateral_alpha = _alpha_power(
        nonlocked.reshape(channels, -1), sampling_rate, nperseg
    )
    return {
        "specificity": specificity,
        "alpha_collateral_fraction": (
            collateral_alpha / raw_alpha if raw_alpha else None
        ),
        "locked_removed_uv": float(np.median(locked_rms)),
        "collateral_uv": float(
            np.median(np.sqrt(np.maximum(total_rms**2 - locked_rms**2, 0.0)))
        ),
    }


def shared_detector_provenance(
    provenances: Mapping[str, CorrectionProvenance],
) -> CorrectionProvenance | None:
    iterator = iter(provenances.items())
    first = next(iterator, None)
    if first is None:
        return None
    reference_arm, reference = first
    for arm, candidate in iterator:
        matches = (
            np.array_equal(reference.peak_samples, candidate.peak_samples)
            and reference.delay_seconds == candidate.delay_seconds
            and reference.window_seconds == candidate.window_seconds
            and reference.gap_fraction == candidate.gap_fraction
        )
        if not matches:
            raise ValueError(
                "inconsistent detector provenance between "
                f"{reference_arm} and {arm}"
            )
    return reference


def load_shared_detector_provenance(
    vhdr_by_arm: Mapping[str, Path],
) -> CorrectionProvenance | None:
    provenances: dict[str, CorrectionProvenance] = {}
    for arm, vhdr in vhdr_by_arm.items():
        provenance = load_correction_provenance(vhdr)
        if provenance is None:
            raise FileNotFoundError(
                f"missing detector provenance for {arm}: "
                f"{vhdr.with_suffix('.bcg.json')}"
            )
        provenances[arm] = provenance
    return shared_detector_provenance(provenances)


def median_locked_ratio(
    raw,
    cleaned,
    *,
    peak_samples: npt.NDArray[np.integer],
    delay_seconds: float,
    window_seconds: tuple[float, float],
) -> float | None:
    if "ECG" not in raw.ch_names:
        return None
    ecg_index = raw.ch_names.index("ECG")
    sampling_rate = float(raw.info["sfreq"])
    locked_window = (
        delay_seconds + window_seconds[0],
        delay_seconds + window_seconds[1],
    )
    before = delay_estimation_eeg(
        raw.get_data(),
        raw.ch_names,
        ecg_channel_index=ecg_index,
    ) * 1e6
    after = delay_estimation_eeg(
        cleaned.get_data(),
        cleaned.ch_names,
        ecg_channel_index=ecg_index,
    ) * 1e6
    before_rms = cardiac_locked_rms(
        before,
        peak_samples,
        sampling_rate_hz=sampling_rate,
        window_seconds=locked_window,
    )
    after_rms = cardiac_locked_rms(
        after,
        peak_samples,
        sampling_rate_hz=sampling_rate,
        window_seconds=locked_window,
    )
    before_median = float(np.median(before_rms))
    if before_median == 0.0:
        return None
    return float(np.median(after_rms) / before_median)
