"""Figures written by `bcgnet run`. Raw vs BCGNet only — no AAS overlay."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy.signal import welch


def plot_before_after_psd(
    before: mne.io.BaseRaw,
    after: mne.io.BaseRaw,
    *,
    title: str,
    output: Path,
    max_hz: float = 100.0,
) -> None:
    """Mean EEG PSD of FASTR input vs BCGNet-cleaned output."""
    plt.figure(figsize=(6, 6))
    plt.title(title)
    for raw, style, label in (
        (before, "C1-", "Before"),
        (after, "C3--", "After"),
    ):
        freqs, pxx = _mean_eeg_psd(raw, max_hz=max_hz)
        plt.semilogy(freqs, pxx, style, label=label)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel(r"PSD ($\mu V^2/Hz)$")
    plt.xlim(0, max_hz)
    plt.legend(loc="upper right")
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, format="png")
    plt.close()


def _mean_eeg_psd(
    raw: mne.io.BaseRaw, *, max_hz: float
) -> tuple[np.ndarray, np.ndarray]:
    names = raw.ch_names
    picks = (
        [i for i, name in enumerate(names) if name != "ECG"]
        if "ECG" in names
        else list(range(len(names)))
    )
    data = raw.get_data(picks=picks) * 1e6
    fs = float(raw.info["sfreq"])
    nperseg = min(int(fs * 3), data.shape[1])
    freqs, pxx = welch(data, fs=fs, nperseg=nperseg, axis=1)
    keep = freqs <= max_hz
    return freqs[keep], np.mean(pxx[:, keep], axis=0)
