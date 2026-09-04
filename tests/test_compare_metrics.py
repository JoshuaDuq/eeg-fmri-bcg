import json
from pathlib import Path

import mne
import numpy as np
import pytest

from bcg_correction.evaluation import EvaluationSettings
from bcg_correction.provenance import (
    CorrectionProvenance,
    load_correction_provenance,
)
from bcgnet.compare.pairs import RecordingSet
from bcgnet.compare.plots import metrics_row
from bcgnet.compare.qc import (
    alpha_peak_height,
    shared_detector_provenance,
)

_EVALUATION = EvaluationSettings((2, 5, 10, 20), 8)


def test_alpha_peak_height_picks_the_ten_hz_bin() -> None:
    freqs = np.arange(0.0, 40.0, 0.5)
    pxx = np.ones_like(freqs)
    pxx[freqs == 10.0] = 8.0
    pxx[freqs == 12.0] = 3.0
    assert alpha_peak_height(freqs, pxx) == pytest.approx(8.0)


def test_detector_peaks_load_from_a_pca_obs_provenance_file(
    tmp_path: Path,
) -> None:
    """Both bounded arms write ``<stem>.bcg.json``; neither name is special."""
    vhdr = tmp_path / "BaselineEEG_sub0000_fastr_pcaobs.vhdr"
    vhdr.write_text("x", encoding="utf-8")
    (tmp_path / "BaselineEEG_sub0000_fastr_pcaobs.bcg.json").write_text(
        json.dumps(
            {
                "peak_samples": [100, 900, 1700],
                "ecg_to_bcg_delay_seconds": 0.19,
                "window_seconds": [-0.2, 0.7],
                "rr_gap_fraction": 0.03,
            }
        ),
        encoding="utf-8",
    )
    loaded = load_correction_provenance(vhdr)
    assert loaded is not None
    assert loaded.peak_samples.tolist() == [100, 900, 1700]
    assert loaded.delay_seconds == pytest.approx(0.19)
    assert loaded.window_seconds == pytest.approx((-0.2, 0.7))
    assert loaded.gap_fraction == pytest.approx(0.03)


def test_detector_peaks_are_absent_without_a_provenance_file(
    tmp_path: Path,
) -> None:
    vhdr = tmp_path / "BaselineEEG_sub0000_fastr_aas.vhdr"
    vhdr.write_text("x", encoding="utf-8")
    assert load_correction_provenance(vhdr) is None


def test_detector_provenance_requires_scoring_fields(tmp_path: Path) -> None:
    vhdr = tmp_path / "BaselineEEG_sub0000_fastr_aas.vhdr"
    vhdr.write_text("x", encoding="utf-8")
    vhdr.with_suffix(".bcg.json").write_text(
        json.dumps(
            {
                "peak_samples": [100, 900, 1700],
                "ecg_to_bcg_delay_seconds": 0.19,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="window_seconds"):
        load_correction_provenance(vhdr)


def test_detector_provenance_must_match_across_bounded_arms(
    tmp_path: Path,
) -> None:
    first = CorrectionProvenance(
        peak_samples=np.asarray([100, 900, 1700]),
        delay_seconds=0.19,
        window_seconds=(-0.2, 0.7),
        gap_fraction=0.0,
    )
    second = CorrectionProvenance(
        peak_samples=np.asarray([100, 900, 1701]),
        delay_seconds=0.19,
        window_seconds=(-0.2, 0.7),
        gap_fraction=0.0,
    )

    with pytest.raises(ValueError, match="inconsistent detector provenance"):
        shared_detector_provenance({"aas": first, "pca_obs": second})


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
        profiles={},
        max_hz=30.0,
        evaluation=_EVALUATION,
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
    only_aas = metrics_row(
        recording,
        {"Raw": _raw(), "AAS": _raw(0.5)},
        profiles={},
        max_hz=30.0,
        evaluation=_EVALUATION,
    )
    only_net = metrics_row(
        recording,
        {"Raw": _raw(), "BCGNet": _raw(0.5)},
        profiles={},
        max_hz=30.0,
        evaluation=_EVALUATION,
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
        profiles={},
        max_hz=30.0,
        evaluation=_EVALUATION,
    )
    ratios = [
        row["delta_aas_ratio"],
        row["delta_pca_obs_ratio"],
        row["delta_bcgnet_ratio"],
    ]
    assert ratios == sorted(ratios, reverse=True)
    assert len(set(ratios)) == 3


def test_profile_computation_errors_surface(tmp_path: Path, monkeypatch) -> None:
    from bcgnet.compare import plots

    provenance = CorrectionProvenance(
        peak_samples=np.asarray([100, 900, 1700]),
        delay_seconds=0.19,
        window_seconds=(-0.2, 0.7),
        gap_fraction=0.0,
    )
    monkeypatch.setattr(plots, "detector_provenance", lambda recording: provenance)

    def fail(*args, **kwargs):
        raise RuntimeError("invalid profile input")

    monkeypatch.setattr(plots, "compute_correction_profile", fail)

    with pytest.raises(RuntimeError, match="invalid profile input"):
        plots.measure_recording(
            _recording(tmp_path),
            {"Raw": _raw(), "AAS": _raw(0.5)},
            _EVALUATION,
        )
