"""Recordings must be labelled from their filenames, never from their position."""

from pathlib import Path

import pytest

from bcgnet.discovery import (
    DEFAULT_RUN_PATTERN,
    Recording,
    iter_subjects,
    label_recordings,
)

_HEADER = "Brain Vision Data Exchange Header File Version 1.0\n" + ("x" * 120)


def _touch(folder: Path, *names: str) -> list[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in names:
        path = folder / name
        path.write_text(_HEADER, encoding="utf-8")
        paths.append(path)
    return paths


def _labels(recordings: list[Recording]) -> list[str]:
    return [recording.label for recording in recordings]


def test_run_number_comes_from_the_filename_not_the_listing_position():
    """sub-0004 has a baseline first, so position would shift every run by one."""
    paths = [
        Path("BaselineEEG_sub0004_2026-05-11_10h47.21.755_fastr_bcgnet.vhdr"),
        Path("ThermalPainEEGFMRI_run1_sub0004_2026-05-11_10h59.27.891_fastr.vhdr"),
        Path("ThermalPainEEGFMRI_run6_sub0004_2026-05-11_11h49.33.381_fastr.vhdr"),
    ]
    recordings = label_recordings(paths)
    assert [recording.run for recording in recordings] == [None, 1, 6]
    assert _labels(recordings) == ["BaselineEEG", "run1", "run6"]


def test_recording_without_a_run_token_is_not_a_run():
    (baseline,) = label_recordings(
        [Path("BaselineEEG_sub0004_2026-05-11_10h47.21.755_fastr_bcgnet.vhdr")]
    )
    assert baseline.run is None
    assert baseline.is_run is False


def test_missing_run_one_is_not_invented():
    """sub-0000 deliberately has no run 1; nothing may claim otherwise."""
    paths = [
        Path("BaselineEEG_sub0000_2026-02-09_10h56.55.966_fastr.vhdr"),
        *(
            Path(f"ThermalPainEEGFMRI_run{n}_sub0000_2026-02-09_11h.vhdr")
            for n in (2, 3, 4, 5, 6)
        ),
    ]
    recordings = label_recordings(paths)
    assert [recording.run for recording in recordings] == [None, 2, 3, 4, 5, 6]
    assert "run1" not in _labels(recordings)


def test_runs_sort_numerically_with_non_runs_first():
    paths = [
        Path("Task_run10_sub0001.vhdr"),
        Path("Task_run2_sub0001.vhdr"),
        Path("Rest_sub0001.vhdr"),
    ]
    assert _labels(label_recordings(paths)) == ["Rest", "run2", "run10"]


@pytest.mark.parametrize(
    ("name", "run"),
    [
        ("sub-01_task-rest_run-02_bold.vhdr", 2),
        ("study_RUN_7_sub0001.vhdr", 7),
        ("acq.run.3.sub0001.vhdr", 3),
        ("Task_run08_sub0001.vhdr", 8),
    ],
)
def test_default_pattern_accepts_common_run_spellings(name, run):
    (recording,) = label_recordings([Path(name)])
    assert recording.run == run


@pytest.mark.parametrize(
    "name",
    [
        # "fastr" and "prerun" both contain letters before the token.
        "BaselineEEG_sub0004_2026-05-11_10h47.21.755_fastr_bcgnet.vhdr",
        "Task_prerun2_sub0001.vhdr",
    ],
)
def test_default_pattern_does_not_match_run_inside_a_word(name):
    (recording,) = label_recordings([Path(name)])
    assert recording.run is None


def test_a_study_can_supply_its_own_run_pattern():
    """Nothing in the code may assume this cohort's task names."""
    paths = [
        Path("acquisition_S03_participant.vhdr"),
        Path("acquisition_rest_participant.vhdr"),
    ]
    recordings = label_recordings(paths, run_pattern=r"_S(\d+)_")
    assert [recording.run for recording in recordings] == [None, 3]
    assert _labels(recordings) == ["acquisition", "run3"]


def test_colliding_non_run_labels_stay_unique():
    paths = [
        Path("Rest_sub0001_first.vhdr"),
        Path("Rest_sub0001_second.vhdr"),
    ]
    assert _labels(label_recordings(paths)) == ["Rest", "Rest_2"]


def test_iter_subjects_yields_labelled_recordings(tmp_path):
    _touch(
        tmp_path / "sub-0000",
        "BaselineEEG_sub0000_fastr.vhdr",
        "ThermalPainEEGFMRI_run2_sub0000_fastr.vhdr",
    )
    ((bids_id, str_sub, recordings),) = iter_subjects(tmp_path)
    assert (bids_id, str_sub) == ("sub-0000", "sub0000")
    assert _labels(recordings) == ["BaselineEEG", "run2"]
    assert recordings[1].path.name == "ThermalPainEEGFMRI_run2_sub0000_fastr.vhdr"
    assert recordings[1].stem == "ThermalPainEEGFMRI_run2_sub0000_fastr"


def test_iter_subjects_honours_a_custom_run_pattern(tmp_path):
    _touch(tmp_path / "sub-0000", "acquisition_S03_sub0000.vhdr")
    ((_, _, recordings),) = iter_subjects(tmp_path, run_pattern=r"_S(\d+)_")
    assert recordings[0].run == 3


def test_default_run_pattern_is_exported_for_configuration():
    assert "run" in DEFAULT_RUN_PATTERN
