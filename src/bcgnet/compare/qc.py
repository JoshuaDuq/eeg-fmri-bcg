"""QC numbers per arm: leftover bands, locked residual, alpha peak."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import numpy.typing as npt

from bcg_correction.metrics import cardiac_locked_rms, delay_estimation_eeg

ALPHA_PEAK_BAND = (8.0, 13.0)
ALPHA_PEAK_COLLAPSE = 0.5


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


def load_detector_peaks(vhdr: Path) -> tuple[np.ndarray, float] | None:
    """Return (peak_samples, applied delay) from a bounded arm's provenance.

    Both bounded arms run the same independent detector and both write
    ``<stem>.bcg.json``, so either one supplies the R train used to score
    every arm's heartbeat-locked residual.
    """
    path = vhdr.parent / f"{vhdr.stem}.bcg.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    peaks = payload.get("peak_samples")
    if not peaks:
        return None
    delay = float(payload.get("ecg_to_bcg_delay_seconds") or 0.0)
    return np.asarray(peaks, dtype=np.int64), delay


def median_locked_ratio(
    raw,
    cleaned,
    *,
    peak_samples: npt.NDArray[np.integer],
    delay_seconds: float,
    window_seconds: tuple[float, float],
) -> float | None:
    """Heartbeat-locked residual of cleaned vs raw (posterior, ECG-regressed)."""
    if "ECG" not in raw.ch_names:
        return None
    ecg_index = raw.ch_names.index("ECG")
    sampling_rate = float(raw.info["sfreq"])
    locked_window = (
        delay_seconds + window_seconds[0],
        delay_seconds + window_seconds[1],
    )
    try:
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
    except Exception:
        return None
    before_median = float(np.median(before_rms))
    if before_median == 0.0:
        return None
    return float(np.median(after_rms) / before_median)
