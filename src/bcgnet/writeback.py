"""Apply a low-rate BCG estimate to original-rate EEG."""

from __future__ import annotations

import mne
import numpy as np
import numpy.typing as npt
from scipy.interpolate import PchipInterpolator


def interpolate_bcg(
    bcg_eeg: npt.NDArray[np.floating],
    source_times: npt.NDArray[np.floating],
    dest_times: npt.NDArray[np.floating],
) -> npt.NDArray[np.float64]:
    bcg = np.asarray(bcg_eeg, dtype=np.float64)
    src = np.asarray(source_times, dtype=np.float64)
    dest = np.asarray(dest_times, dtype=np.float64)
    if bcg.ndim != 2:
        raise ValueError("bcg_eeg must have shape (n_eeg, n_times)")
    if src.ndim != 1 or dest.ndim != 1:
        raise ValueError("time vectors must be one-dimensional")
    if bcg.shape[1] != src.size:
        raise ValueError("bcg_eeg time axis must match source_times")
    interpolator = PchipInterpolator(src, bcg, axis=1, extrapolate=True)
    return interpolator(dest)


def unstandardized_bcg(
    predicted_bcg_standardized: npt.NDArray[np.floating],
    eeg_std: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    bcg = np.asarray(predicted_bcg_standardized, dtype=np.float64)
    std = np.asarray(eeg_std, dtype=np.float64)
    if bcg.ndim != 2:
        raise ValueError("predicted BCG must have shape (n_eeg, n_times)")
    if std.ndim != 1 or std.size != bcg.shape[0]:
        raise ValueError("eeg_std must contain one scale per EEG channel")
    return bcg * std[:, np.newaxis]


def subtract_interpolated_bcg(
    original: mne.io.BaseRaw,
    bcg_eeg: npt.NDArray[np.floating],
    bcg_times: npt.NDArray[np.floating],
    *,
    ecg_channel: str = "ECG",
) -> mne.io.RawArray:
    names = list(original.ch_names)
    if ecg_channel not in names:
        raise ValueError(f"ECG channel {ecg_channel!r} is not in {names}")
    ecg_index = names.index(ecg_channel)
    eeg_indices = [index for index in range(len(names)) if index != ecg_index]
    bcg = np.asarray(bcg_eeg, dtype=np.float64)
    if bcg.shape[0] != len(eeg_indices):
        raise ValueError(
            f"BCG has {bcg.shape[0]} channels, expected {len(eeg_indices)} EEG"
        )
    bcg_hi = interpolate_bcg(bcg, bcg_times, original.times)
    cleaned = original.get_data().astype(np.float64, copy=True)
    cleaned[np.asarray(eeg_indices, dtype=int), :] -= bcg_hi
    return mne.io.RawArray(cleaned, original.info, verbose=False)
