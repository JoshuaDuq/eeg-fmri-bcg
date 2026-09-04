"""Pairing, FASTR alignment, and inclusion rules for the simulator experiment."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@dataclass(frozen=True, slots=True)
class _RawShape:
    ch_names: tuple[str, ...]
    sampling_rate: float
    n_times: int

    @property
    def info(self) -> dict[str, float]:
        return {"sfreq": self.sampling_rate}


def test_scanner_alignment_requires_same_grid_channels_and_events() -> None:
    from experiments.mri_simulator_preservation.run import _validate_scanner_alignment

    reference = _RawShape(("Pz", "Oz", "ECG"), 1000.0, 20_000)
    candidate = _RawShape(("Pz", "Oz", "ECG"), 1000.0, 20_000)
    events = np.array([1000, 5000, 9000])

    _validate_scanner_alignment(reference, events, candidate, events, "aas")

    with pytest.raises(ValueError, match="aas event samples differ from FASTR"):
        _validate_scanner_alignment(
            reference,
            events,
            candidate,
            np.array([1000, 5001, 9000]),
            "aas",
        )


@pytest.mark.parametrize(
    ("candidate", "match"),
    [
        (_RawShape(("Oz", "Pz", "ECG"), 1000.0, 20_000), "channel order"),
        (_RawShape(("Pz", "Oz", "ECG"), 500.0, 20_000), "sampling rate"),
        (_RawShape(("Pz", "Oz", "ECG"), 1000.0, 19_000), "sample count"),
    ],
)
def test_scanner_alignment_rejects_grid_mismatches(candidate, match) -> None:
    from experiments.mri_simulator_preservation.run import _validate_scanner_alignment

    reference = _RawShape(("Pz", "Oz", "ECG"), 1000.0, 20_000)
    events = np.array([1000, 5000, 9000])
    with pytest.raises(ValueError, match=match):
        _validate_scanner_alignment(reference, events, candidate, events, "aas")


def test_find_thermal_vhdr_rejects_duplicate_scanner_files(tmp_path: Path) -> None:
    from experiments.mri_simulator_preservation.run import _find_thermal_vhdr

    folder = tmp_path / "sub-0019"
    folder.mkdir()
    (folder / "ThermalPainEEGFMRI_run1_a_fastr.vhdr").write_text("x" * 120)
    (folder / "ThermalPainEEGFMRI_run1_b_fastr.vhdr").write_text("y" * 80)
    with pytest.raises(ValueError, match="more than one"):
        _find_thermal_vhdr(folder, 1)


def test_load_compare_summary_requires_the_file_and_cardiac_columns(
    tmp_path: Path,
) -> None:
    from experiments.mri_simulator_preservation.run import _load_compare_summary

    missing = tmp_path / "absent.csv"
    with pytest.raises(FileNotFoundError, match="missing comparison summary"):
        _load_compare_summary(missing, cardiac_template="local_5_{method}")

    path = tmp_path / "compare.csv"
    path.write_text("bids_id,run\nsub-0019,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing columns"):
        _load_compare_summary(path, cardiac_template="local_5_{method}")
