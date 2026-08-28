from pathlib import Path

import mne
import numpy as np

from bcgnet.compare.plots import plot_epoch
from bcgnet.figures import plot_before_after_psd


def _raw(sfreq: float = 1000.0, n: int = 4000) -> mne.io.RawArray:
    t = np.arange(n) / sfreq
    eeg = 20e-6 * np.sin(2 * np.pi * 1.2 * t)
    ecg = 1e-3 * np.sin(2 * np.pi * 1.0 * t)
    info = mne.create_info(["Cz", "ECG"], sfreq, ch_types=["eeg", "ecg"])
    return mne.io.RawArray(np.vstack([eeg, ecg]), info, verbose="ERROR")


def test_plot_epoch_writes_a_png(tmp_path: Path) -> None:
    raw = _raw()
    cleaned = _raw()
    cleaned._data[0] *= 0.8
    output = tmp_path / "epoch.png"
    plot_epoch(
        {"Raw": raw, "AAS": cleaned, "BCGNet": cleaned},
        channel="Cz",
        start=0.0,
        duration=3.0,
        title="sub-0000 Cz run 4",
        output=output,
    )
    assert output.is_file()
    assert output.stat().st_size > 1000


def test_plot_before_after_psd_writes_a_png(tmp_path: Path) -> None:
    output = tmp_path / "psd_run1_avg.png"
    plot_before_after_psd(
        _raw(),
        _raw(),
        title="sub-0000 run 1 before vs after",
        output=output,
    )
    assert output.is_file()
    assert output.stat().st_size > 1000
