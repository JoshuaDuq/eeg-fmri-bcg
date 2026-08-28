import mne
import numpy as np
import pytest
from scipy.signal import welch

from bcgnet.writeback import subtract_interpolated_bcg, unstandardized_bcg


def _raw(data: np.ndarray, sfreq: float, names: list[str]) -> mne.io.RawArray:
    types = ["ecg" if name == "ECG" else "eeg" for name in names]
    info = mne.create_info(names, sfreq, ch_types=types)
    return mne.io.RawArray(data, info, verbose="ERROR")


def test_subtract_interpolated_bcg_keeps_line_noise_and_original_ecg() -> None:
    sfreq_hi = 1000.0
    sfreq_lo = 100.0
    duration = 2.0
    n_hi = int(sfreq_hi * duration)
    n_lo = int(sfreq_lo * duration)
    t_hi = np.arange(n_hi) / sfreq_hi
    t_lo = np.arange(n_lo) / sfreq_lo
    bcg = 20e-6 * np.sin(2 * np.pi * 1.2 * t_hi)
    line = 5e-6 * np.sin(2 * np.pi * 60.0 * t_hi)
    eeg = bcg + line
    ecg = np.zeros(n_hi)
    ecg[200] = 1e-3
    original = _raw(np.vstack([eeg, eeg, ecg]), sfreq_hi, ["Cz", "Pz", "ECG"])
    bcg_lo = np.vstack(
        [
            20e-6 * np.sin(2 * np.pi * 1.2 * t_lo),
            20e-6 * np.sin(2 * np.pi * 1.2 * t_lo),
        ]
    )

    cleaned = subtract_interpolated_bcg(
        original,
        bcg_lo,
        t_lo,
        ecg_channel="ECG",
    )

    assert cleaned.info["sfreq"] == sfreq_hi
    assert np.array_equal(cleaned.get_data()[2], original.get_data()[2])
    residual = cleaned.get_data()[0]
    freqs, pxx = welch(residual, fs=sfreq_hi, nperseg=min(1000, residual.size))
    orig_f, orig_p = welch(eeg, fs=sfreq_hi, nperseg=min(1000, eeg.size))

    def band(values_f, values_p, low, high):
        mask = (values_f >= low) & (values_f <= high)
        return float(np.sum(values_p[mask]))

    assert band(freqs, pxx, 55, 65) == pytest.approx(
        band(orig_f, orig_p, 55, 65), rel=0.15
    )
    assert band(freqs, pxx, 0.5, 2.0) < 0.2 * band(orig_f, orig_p, 0.5, 2.0)


def test_construct_epoch_events_accepts_float_epoch_length() -> None:
    from bcgnet.runtime import prepare_vendor_imports

    prepare_vendor_imports()
    from dataset.default_dataset import Dataset

    events, tmax = Dataset._construct_epoch_events(10_000, 100, 3.0)
    assert events.shape[0] >= 1
    assert tmax == pytest.approx(3.0 - 1 / 100)


def test_unstandardized_bcg_scales_without_adding_channel_means() -> None:
    predicted = np.array([[1.0, -1.0], [0.5, -0.5]])
    scaled = unstandardized_bcg(predicted, [2.0, 4.0])
    assert scaled == pytest.approx(np.array([[2.0, -2.0], [2.0, -2.0]]))
