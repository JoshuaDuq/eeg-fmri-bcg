"""The bounded comparator batch must cover every recording, in parallel or not.

Each recording is an independent correction, so the batch is free to spread
them across processes. What must not change is the result: one row per
recording, every output written, whatever the worker count.
"""

from pathlib import Path

import numpy as np
import pytest
from pybv import write_brainvision

from bcg_correction.bcg_config import DetectorConfig
from bcg_correction.brainvision import (
    BrainVisionMarker,
    write_brainvision_markers,
)
from bcgnet.compare.arms import AAS
from bcgstudy.correction_batch import (
    CorrectionSettings,
    run_correction_batch,
    write_aggregate_reports,
)

_DETECTOR = DetectorConfig(
    ecg_channel="ECG",
    preprocessing_band_hz=(0.5, 10.0),
    teager_emphasis_hz=10.0,
    teager_smoothing_seconds=0.028,
    template_window_seconds=(-0.2, 0.4),
    minimum_rr_seconds=0.4,
    maximum_rr_seconds=2.0,
    candidate_refractory_seconds=0.25,
    candidate_prominence_mad=2.0,
    correlation_threshold=0.5,
    refinement_iterations=2,
)

_SETTINGS = CorrectionSettings(
    window_seconds=(-0.1, 0.2),
    ecg_to_bcg_delay_seconds=0.21,
    aas_neighbor_count=2,
    pca_obs_components=1,
    cross_fit_fold_count=2,
    maximum_residual_ratio=0.75,
    residual_floor_uv=0.0,
    maximum_gap_fraction=0.05,
    overwrite=False,
    detector=_DETECTOR,
)


def _write_recording(folder: Path, stem: str) -> Path:
    """One short synthetic FASTR recording with a clean R train."""
    sampling_rate_hz = 1_000.0
    sample_count = 6_000
    samples = np.arange(sample_count, dtype=float)
    peak_samples = np.array([800, 1_650, 2_530, 3_440, 4_370, 5_310])
    ecg = np.zeros(sample_count, dtype=float)
    artifact = np.exp(-0.5 * ((samples - 15.0) / 18.0) ** 2)
    for peak in peak_samples:
        ecg += 1e-3 * np.exp(-0.5 * ((samples - peak) / 8.0) ** 2)
    # Three EEG channels, each with its own rhythm and artifact gain, so the
    # cross-channel metrics see the spread they do on a real montage.
    eeg = []
    rhythms = ((173.0, 25e-6), (91.0, 18e-6), (211.0, 31e-6))
    for index, (period, gain) in enumerate(rhythms):
        drift = 2e-7 * (index + 1)
        trace = 2e-6 + drift * np.sin(2.0 * np.pi * samples / period)
        for peak in peak_samples:
            trace = trace + gain * np.roll(artifact, int(peak + 210 - 15))
        eeg.append(trace)
    folder.mkdir(parents=True, exist_ok=True)
    write_brainvision(
        data=np.vstack((*eeg, ecg)),
        sfreq=sampling_rate_hz,
        ch_names=["O1", "Cz", "Pz", "ECG"],
        fname_base=stem,
        folder_out=folder,
        events=[],
        unit="µV",
    )
    marker_path = folder / f"{stem}.vmrk"
    marker_path.unlink()
    write_brainvision_markers(
        marker_path,
        f"{stem}.eeg",
        (
            BrainVisionMarker("New Segment", "", 1, 1, 0),
            BrainVisionMarker("Volume", "V  1", 101, 1, 0),
        ),
    )
    return folder / f"{stem}.vhdr"


def _fastr_root(tmp_path: Path, *, subjects: int, runs: int) -> Path:
    root = tmp_path / "fastr"
    for subject in range(subjects):
        folder = root / f"sub-{subject:04d}"
        for run in range(1, runs + 1):
            _write_recording(folder, f"task_run{run}_sub{subject:04d}_fastr")
    return root


def test_parallel_batch_corrects_every_recording(tmp_path: Path) -> None:
    """workers=2 must still produce one ok row and one output per recording."""
    fastr_root = _fastr_root(tmp_path, subjects=2, runs=2)
    output_root = tmp_path / "aas"

    rows = run_correction_batch(
        fastr_root=fastr_root,
        output_root=output_root,
        arm=AAS,
        settings=_SETTINGS,
        workers=2,
    )

    assert len(rows) == 4
    assert [row["status"] for row in rows] == ["ok"] * 4
    written = sorted(path.name for path in output_root.rglob("*.vhdr"))
    assert len(written) == 4
    assert all(name.endswith(f"_{AAS.suffix}.vhdr") for name in written)


def test_parallel_batch_matches_serial_batch(tmp_path: Path) -> None:
    """Worker count must not change which recordings the batch reports."""
    fastr_root = _fastr_root(tmp_path, subjects=2, runs=2)

    serial = run_correction_batch(
        fastr_root=fastr_root,
        output_root=tmp_path / "serial",
        arm=AAS,
        settings=_SETTINGS,
        workers=1,
    )
    parallel = run_correction_batch(
        fastr_root=fastr_root,
        output_root=tmp_path / "parallel",
        arm=AAS,
        settings=_SETTINGS,
        workers=4,
    )

    assert [Path(row["input"]).name for row in serial] == [
        Path(row["input"]).name for row in parallel
    ]
    assert [row["marker_count"] for row in serial] == [
        row["marker_count"] for row in parallel
    ]


def test_aggregate_reports_reject_corrupt_profiles(tmp_path: Path) -> None:
    subject = tmp_path / "sub-0000"
    subject.mkdir()
    np.savez_compressed(subject / "run_profile.npz", method=np.asarray("aas"))

    with pytest.raises(ValueError, match="profile schema"):
        write_aggregate_reports(tmp_path, AAS)
