"""GitHub-style Raw / AAS / BCGNet overlays for one recording."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy.signal import welch

from .pairs import RecordingTriple
from .qc import (
    alpha_peak_height,
    load_aas_peaks,
    median_locked_ratio,
    method_qc_flags,
)

CLEAN_METHODS = ("AAS", "BCGNet")


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


def plot_psd(
    traces: dict[str, mne.io.BaseRaw],
    *,
    title: str,
    output: Path,
    max_hz: float,
) -> None:
    styles = {
        "Raw": ("C1-", "Raw"),
        "AAS": ("C2--", "AAS"),
        "BCGNet": ("C3--", "BCGNet"),
    }
    plt.figure(figsize=(6, 6))
    plt.title(title)
    for key, (style, label) in styles.items():
        if key not in traces:
            continue
        freqs, pxx = mean_eeg_psd(traces[key], max_hz=max_hz)
        plt.semilogy(freqs, pxx, style, label=label)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel(r"PSD ($\mu V^2/Hz)$")
    plt.xlim(0, max_hz)
    plt.legend(loc="upper right")
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, format="png")
    plt.close()


def plot_epoch(
    traces: dict[str, mne.io.BaseRaw],
    *,
    channel: str,
    start: float,
    duration: float,
    title: str,
    output: Path,
) -> None:
    raw = traces["Raw"]
    if channel not in raw.ch_names:
        raise ValueError(f"channel {channel!r} is not in {raw.ch_names}")
    fs = float(raw.info["sfreq"])
    start_sample = int(start * fs)
    stop_sample = start_sample + int(duration * fs)
    if stop_sample > raw.n_times:
        start_sample = 0
        stop_sample = min(int(duration * fs), raw.n_times)
    t = np.arange(stop_sample - start_sample) / fs
    ch = raw.ch_names.index(channel)

    x_max = float(t[-1]) * 1.05 if t.size else duration
    plt.figure(figsize=(8, 10))
    plt.suptitle(title, fontweight="bold")

    plt.subplot(311)
    plt.title("Original ECG")
    if "ECG" in raw.ch_names:
        ecg = raw.get_data(picks=["ECG"])[0, start_sample:stop_sample] * 1e6
        plt.plot(t, ecg, "C0")
    plt.xlabel("Time (s)")
    plt.ylabel(r"Amplitude ($\mu$V)")
    plt.xlim([0, x_max])

    plt.subplot(312)
    plt.title("BCGNet-predicted BCG")
    raw_ch = raw.get_data()[ch, start_sample:stop_sample] * 1e6
    if "BCGNet" in traces:
        n = min(stop_sample, traces["BCGNet"].n_times)
        cleaned = traces["BCGNet"].get_data()[ch, start_sample:n] * 1e6
        predicted = raw_ch[: cleaned.size] - cleaned
        plt.plot(t[: predicted.size], predicted, "C4")
    plt.xlabel("Time (s)")
    plt.ylabel(r"Amplitude ($\mu$V)")
    plt.xlim([0, x_max])

    plt.subplot(313)
    plt.title("Raw and Cleaned Data")
    plt.plot(t, raw_ch, "C1", label="Raw")
    overlay = (("AAS", "C2"), ("BCGNet", "C3"))
    for name, color in overlay:
        if name not in traces:
            continue
        n = min(stop_sample, traces[name].n_times)
        if n > start_sample:
            plt.plot(
                t[: n - start_sample],
                traces[name].get_data()[ch, start_sample:n] * 1e6,
                color,
                label=name,
            )
    plt.xlabel("Time (s)")
    plt.ylabel(r"Amplitude ($\mu$V)")
    plt.xlim([0, x_max])
    plt.legend(loc="upper right", frameon=False)
    # Same spacing as the vendor epoch figures.
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, format="png")
    plt.close()


def band_power(freqs: np.ndarray, pxx: np.ndarray, low: float, high: float) -> float:
    mask = (freqs >= low) & (freqs <= high)
    return float(np.sum(pxx[mask]))


def metrics_row(
    triple: RecordingTriple,
    traces: dict[str, mne.io.BaseRaw],
    *,
    max_hz: float,
    window_seconds: tuple[float, float] = (-0.2, 0.7),
) -> dict[str, object]:
    row: dict[str, object] = {
        "bids_id": triple.bids_id,
        "stem": triple.stem,
        "idx_run": triple.idx_run,
        "has_aas": "AAS" in traces,
        "has_bcgnet": "BCGNet" in traces,
    }
    psds = {
        name: mean_eeg_psd(raw, max_hz=max_hz) for name, raw in traces.items()
    }
    raw_f, raw_p = psds["Raw"]
    raw_rms = float(np.sqrt(np.mean(np.square(traces["Raw"].get_data() * 1e6))))
    row["rms_raw"] = raw_rms
    bands = {
        "delta": (0.5, 4.0),
        "theta": (4.0, 8.0),
        "alpha": (8.0, 13.0),
    }
    remaining: dict[str, float | None] = {}
    for band, (low, high) in bands.items():
        raw_band = band_power(raw_f, raw_p, low, high)
        row[f"{band}_raw"] = raw_band
        for method in CLEAN_METHODS:
            if method not in psds:
                continue
            freqs, pxx = psds[method]
            value = band_power(freqs, pxx, low, high)
            row[f"{band}_{method.lower()}"] = value
            ratio = value / raw_band if raw_band else None
            row[f"{band}_{method.lower()}_ratio"] = ratio
            if method == "BCGNet":
                remaining[band] = ratio
    alpha_net = None
    for method in CLEAN_METHODS:
        if method not in traces:
            continue
        data = traces[method].get_data() * 1e6
        row[f"rms_{method.lower()}"] = float(np.sqrt(np.mean(np.square(data))))
        peak = alpha_peak_height(*psds[method])
        row[f"alpha_peak_{method.lower()}"] = peak
        if method == "BCGNet":
            alpha_net = peak
    alpha_raw = alpha_peak_height(raw_f, raw_p)
    row["alpha_peak_raw"] = alpha_raw

    peaks = (
        load_aas_peaks(triple.aas_vhdr) if triple.aas_vhdr is not None else None
    )
    locked: dict[str, float | None] = {name.lower(): None for name in CLEAN_METHODS}
    if peaks is not None:
        peak_samples, delay = peaks
        for method in CLEAN_METHODS:
            if method not in traces:
                continue
            locked[method.lower()] = median_locked_ratio(
                traces["Raw"],
                traces[method],
                peak_samples=peak_samples,
                delay_seconds=delay,
                window_seconds=window_seconds,
            )
    row["locked_aas_ratio"] = locked["aas"]
    row["locked_bcgnet_ratio"] = locked["bcgnet"]
    locked_net = locked["bcgnet"]
    flags = method_qc_flags(
        remaining_ratios=remaining,
        locked_ratio=locked_net,
        alpha_peak_raw=alpha_raw,
        alpha_peak_bcgnet=alpha_net,
    )
    row.update(flags)
    return row
