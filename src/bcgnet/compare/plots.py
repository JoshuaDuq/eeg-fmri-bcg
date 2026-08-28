"""Raw vs corrected-arm overlays and the per-recording metrics row."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy.signal import welch

from .arms import BCGNET, CLEAN_ARMS, COMPARATOR_ARMS
from .pairs import RecordingSet
from .qc import (
    alpha_peak_height,
    load_detector_peaks,
    median_locked_ratio,
    method_qc_flags,
)

RAW_LABEL = "Raw"
_RAW_STYLE = "C1-"
_RAW_COLOR = "C1"
# The BCG the network removed, shown as its own subplot rather than an arm.
_PREDICTED_COLOR = "C4"

_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
}


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
    plt.figure(figsize=(6, 6))
    plt.title(title)
    if RAW_LABEL in traces:
        freqs, pxx = mean_eeg_psd(traces[RAW_LABEL], max_hz=max_hz)
        plt.semilogy(freqs, pxx, _RAW_STYLE, label=RAW_LABEL)
    for arm in CLEAN_ARMS:
        if arm.label not in traces:
            continue
        freqs, pxx = mean_eeg_psd(traces[arm.label], max_hz=max_hz)
        plt.semilogy(freqs, pxx, arm.style, label=arm.label)
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
    raw = traces[RAW_LABEL]
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
    if BCGNET.label in traces:
        n = min(stop_sample, traces[BCGNET.label].n_times)
        cleaned = traces[BCGNET.label].get_data()[ch, start_sample:n] * 1e6
        predicted = raw_ch[: cleaned.size] - cleaned
        plt.plot(t[: predicted.size], predicted, _PREDICTED_COLOR)
    plt.xlabel("Time (s)")
    plt.ylabel(r"Amplitude ($\mu$V)")
    plt.xlim([0, x_max])

    plt.subplot(313)
    plt.title("Raw and Cleaned Data")
    plt.plot(t, raw_ch, _RAW_COLOR, label=RAW_LABEL)
    for arm in CLEAN_ARMS:
        if arm.label not in traces:
            continue
        n = min(stop_sample, traces[arm.label].n_times)
        if n > start_sample:
            plt.plot(
                t[: n - start_sample],
                traces[arm.label].get_data()[ch, start_sample:n] * 1e6,
                arm.color,
                label=arm.label,
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


def _detector_peaks(recording: RecordingSet) -> tuple[np.ndarray, float] | None:
    """R train for locked residuals, from whichever bounded arm produced one."""
    for arm in COMPARATOR_ARMS:
        vhdr = recording.cleaned_vhdr.get(arm.key)
        if vhdr is None:
            continue
        peaks = load_detector_peaks(vhdr)
        if peaks is not None:
            return peaks
    return None


def metrics_row(
    recording: RecordingSet,
    traces: dict[str, mne.io.BaseRaw],
    *,
    max_hz: float,
    window_seconds: tuple[float, float] = (-0.2, 0.7),
) -> dict[str, object]:
    """One row of ``compare_summary.csv``.

    Every arm contributes the same columns whether or not it ran, so the summary
    stays a single rectangular table across a cohort with uneven coverage.
    """
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

    peaks = _detector_peaks(recording)
    for arm in CLEAN_ARMS:
        ratio = None
        if peaks is not None and arm.label in traces:
            peak_samples, delay = peaks
            ratio = median_locked_ratio(
                traces[RAW_LABEL],
                traces[arm.label],
                peak_samples=peak_samples,
                delay_seconds=delay,
                window_seconds=window_seconds,
            )
        row[f"locked_{arm.key}_ratio"] = ratio

    row.update(
        method_qc_flags(
            remaining_ratios=remaining,
            locked_ratio=row[f"locked_{BCGNET.key}_ratio"],
            alpha_peak_raw=alpha_raw,
            alpha_peak_bcgnet=alpha_net,
        )
    )
    return row
