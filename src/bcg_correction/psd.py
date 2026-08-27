"""Shared power spectral density diagnostics for corrected recordings."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np

PSD_MAX_FREQUENCY_HZ = 100.0
PSD_FFT_SAMPLES = 2048


def save_psd_plot(
    raw: mne.io.BaseRaw,
    output_path: Path,
    *,
    fmax: float,
    title: str,
    tmin: float,
    tmax: float,
    n_fft: int | None = None,
) -> None:
    """Save an MNE raw PSD figure with spatial channel colors when possible."""
    plot_raw = prepare_psd_raw(raw)
    plot_kwargs = {
        "fmin": 0.0,
        "fmax": fmax,
        "tmin": tmin,
        "tmax": tmax,
        "spatial_colors": plot_raw.get_montage() is not None,
        "show": False,
        "n_jobs": 1,
        "verbose": "ERROR",
    }
    if n_fft is not None:
        plot_kwargs["n_fft"] = n_fft
    with mne.use_log_level("ERROR"):
        figure = mne.viz.plot_raw_psd(plot_raw, **plot_kwargs)
    try:
        for axis in figure.axes:
            if axis.get_xlabel():
                axis.set_xlim(0.0, fmax)
        figure.suptitle(title)
        figure.savefig(output_path, dpi=150, bbox_inches="tight")
    finally:
        plt.close(figure)


def prepare_psd_raw(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """Give PSD plots standard positions when the file has no montage."""
    prepared = raw.copy()
    montage = prepared.get_montage()
    if montage is not None:
        positioned_names = _positioned_channel_names(prepared, montage)
        if len(positioned_names) >= 2:
            prepared.pick(positioned_names)
            return prepared

    standard = mne.channels.make_standard_montage("standard_1020")
    standard_names = {name.casefold() for name in standard.ch_names}
    matched_names = [
        name for name in prepared.ch_names if name.casefold() in standard_names
    ]
    if len(matched_names) >= 2:
        prepared.pick(matched_names)
        prepared.set_montage(
            standard,
            match_case=False,
            on_missing="raise",
            verbose="ERROR",
        )
    return prepared


def _positioned_channel_names(
    raw: mne.io.BaseRaw,
    montage: mne.channels.DigMontage,
) -> list[str]:
    positions = montage.get_positions()["ch_pos"]
    return [
        name
        for name in raw.ch_names
        if name in positions and np.isfinite(positions[name]).all()
    ]
