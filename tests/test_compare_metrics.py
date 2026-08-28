import json
from pathlib import Path

import mne
import numpy as np
import pytest

from bcgnet.compare.pairs import RecordingSet
from bcgnet.compare.plots import metrics_row
from bcgnet.compare.qc import (
    alpha_peak_height,
    load_detector_peaks,
    method_qc_flags,
    remaining_ratio,
)


def test_alpha_peak_height_picks_the_ten_hz_bin() -> None:
    freqs = np.arange(0.0, 40.0, 0.5)
    pxx = np.ones_like(freqs)
    pxx[freqs == 10.0] = 8.0
    pxx[freqs == 12.0] = 3.0
    assert alpha_peak_height(freqs, pxx) == pytest.approx(8.0)


def test_remaining_ratio_flags_added_power() -> None:
    assert remaining_ratio(2.0, 1.0) == pytest.approx(2.0)
    assert remaining_ratio(0.4, 1.0) == pytest.approx(0.4)


def test_qc_flags_prefer_a_comparator_when_bcgnet_adds_power() -> None:
    flags = method_qc_flags(
        remaining_ratios={"delta": 1.6, "theta": 0.5, "alpha": 0.4},
        locked_ratio=0.7,
        alpha_peak_raw=8.0,
        alpha_peak_bcgnet=7.5,
    )
    assert flags["bcgnet_adds_power"] is True
    assert flags["prefer_comparator"] is True
    assert flags["bcgnet_locked_worse_than_raw"] is False


def test_qc_flags_do_not_prefer_a_comparator_only_because_alpha_peak_fell() -> None:
    flags = method_qc_flags(
        remaining_ratios={"delta": 0.4, "theta": 0.3, "alpha": 0.3},
        locked_ratio=0.3,
        alpha_peak_raw=20.0,
        alpha_peak_bcgnet=4.0,
    )
    assert flags["alpha_peak_collapsed"] is True
    assert flags["prefer_comparator"] is False


def test_qc_flags_prefer_a_comparator_when_locked_residual_increases() -> None:
    flags = method_qc_flags(
        remaining_ratios={"delta": 0.6, "theta": 0.4, "alpha": 0.5},
        locked_ratio=1.2,
        alpha_peak_raw=8.0,
        alpha_peak_bcgnet=7.0,
    )
    assert flags["bcgnet_locked_worse_than_raw"] is True
    assert flags["prefer_comparator"] is True


def test_qc_flags_keep_bcgnet_when_harmonics_fall_and_alpha_peak_stays() -> None:
    flags = method_qc_flags(
        remaining_ratios={"delta": 0.6, "theta": 0.3, "alpha": 0.5},
        locked_ratio=0.4,
        alpha_peak_raw=8.0,
        alpha_peak_bcgnet=7.2,
    )
    assert flags["prefer_comparator"] is False
    assert flags["alpha_peak_collapsed"] is False


def test_detector_peaks_load_from_a_pca_obs_provenance_file(
    tmp_path: Path,
) -> None:
    """Both bounded arms write ``<stem>.bcg.json``; neither name is special."""
    vhdr = tmp_path / "BaselineEEG_sub0000_fastr_pcaobs.vhdr"
    vhdr.write_text("x", encoding="utf-8")
    (tmp_path / "BaselineEEG_sub0000_fastr_pcaobs.bcg.json").write_text(
        json.dumps(
            {"peak_samples": [100, 900, 1700], "ecg_to_bcg_delay_seconds": 0.19}
        ),
        encoding="utf-8",
    )
    loaded = load_detector_peaks(vhdr)
    assert loaded is not None
    peaks, delay = loaded
    assert peaks.tolist() == [100, 900, 1700]
    assert delay == pytest.approx(0.19)


def test_detector_peaks_are_absent_without_a_provenance_file(
    tmp_path: Path,
) -> None:
    vhdr = tmp_path / "BaselineEEG_sub0000_fastr_bcg.vhdr"
    vhdr.write_text("x", encoding="utf-8")
    assert load_detector_peaks(vhdr) is None


def _raw(scale: float = 1.0, sfreq: float = 1000.0, n: int = 4000) -> mne.io.RawArray:
    t = np.arange(n) / sfreq
    eeg = scale * 20e-6 * np.sin(2 * np.pi * 1.2 * t)
    ecg = 1e-3 * np.sin(2 * np.pi * 1.0 * t)
    info = mne.create_info(["Cz", "ECG"], sfreq, ch_types=["eeg", "ecg"])
    return mne.io.RawArray(np.vstack([eeg, ecg]), info, verbose="ERROR")


def _recording(tmp_path: Path) -> RecordingSet:
    return RecordingSet(
        bids_id="sub-0000",
        str_sub="sub0000",
        label="BaselineEEG",
        run=None,
        stem="BaselineEEG_sub0000_fastr",
        fastr_vhdr=tmp_path / "BaselineEEG_sub0000_fastr.vhdr",
        cleaned_vhdr={},
    )


def test_metrics_row_reports_pca_obs_under_its_own_columns(
    tmp_path: Path,
) -> None:
    row = metrics_row(
        _recording(tmp_path),
        {"Raw": _raw(), "PCA-OBS": _raw(scale=0.5)},
        max_hz=30.0,
    )
    assert row["has_pca_obs"] is True
    assert row["has_aas"] is False
    assert row["has_bcgnet"] is False
    assert row["delta_pca_obs_ratio"] == pytest.approx(0.25, rel=0.05)
    assert row["delta_aas_ratio"] is None


def test_metrics_row_columns_do_not_depend_on_which_arms_exist(
    tmp_path: Path,
) -> None:
    """compare_summary.csv is one table, so every row needs the same columns."""
    recording = _recording(tmp_path)
    only_aas = metrics_row(recording, {"Raw": _raw(), "AAS": _raw(0.5)}, max_hz=30.0)
    only_net = metrics_row(
        recording, {"Raw": _raw(), "BCGNet": _raw(0.5)}, max_hz=30.0
    )
    assert list(only_aas.keys()) == list(only_net.keys())


def test_metrics_row_keeps_every_arm_separate(tmp_path: Path) -> None:
    row = metrics_row(
        _recording(tmp_path),
        {
            "Raw": _raw(),
            "AAS": _raw(scale=0.5),
            "PCA-OBS": _raw(scale=0.25),
            "BCGNet": _raw(scale=0.125),
        },
        max_hz=30.0,
    )
    ratios = [
        row["delta_aas_ratio"],
        row["delta_pca_obs_ratio"],
        row["delta_bcgnet_ratio"],
    ]
    assert ratios == sorted(ratios, reverse=True)
    assert len(set(ratios)) == 3
