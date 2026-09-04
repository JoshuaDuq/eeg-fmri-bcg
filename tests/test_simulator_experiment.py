"""MRI-simulator preservation experiment: no B0 means scanner BCG is absent."""

from pathlib import Path


def _write_vhdr(path: Path, *, data_file: str, marker_file: str) -> None:
    path.write_text(
        "Brain Vision Data Exchange Header File Version 1.0\n"
        f"DataFile={data_file}\n"
        f"MarkerFile={marker_file}\n",
        encoding="utf-8",
    )


def test_bids_id_from_simulator_folder_name() -> None:
    from bcgstudy.simulator_experiment import bids_id_from_folder

    assert bids_id_from_folder("sub_0010") == "sub-0010"
    assert bids_id_from_folder("sub-0015") == "sub-0015"


def test_discover_keeps_requested_range_and_skips_others(tmp_path: Path) -> None:
    from bcgstudy.simulator_experiment import discover_simulator_recordings

    for name in ("sub_0008", "sub_0010", "sub_0011", "sub_0021"):
        eeg = tmp_path / name / "eeg"
        eeg.mkdir(parents=True)
        stem = f"BaselineEEG_{name.replace('_', '')}_eeg"
        (eeg / f"{stem}.eeg").write_bytes(b"x" * 200)
        (eeg / f"{stem}.vmrk").write_text(
            "Brain Vision Data Exchange Marker File Version 1.0\n"
        )
        _write_vhdr(
            eeg / f"{stem}.vhdr",
            data_file=f"{stem}.eeg",
            marker_file=f"{stem}.vmrk",
        )

    found = discover_simulator_recordings(tmp_path, include_from=10, include_to=21)
    ids = {item.bids_id for item in found}
    assert ids == {"sub-0010", "sub-0011", "sub-0021"}
    assert "sub-0008" not in ids


def test_rewrite_header_sidecars_to_match_the_vhdr_stem(tmp_path: Path) -> None:
    from bcgstudy.simulator_experiment import rewrite_header_sidecars

    vhdr = tmp_path / "BaselineEEG_sub0010_eeg_only.vhdr"
    _write_vhdr(
        vhdr,
        data_file="BaselineEEG_sub0011_eeg_only.eeg",
        marker_file="BaselineEEG_sub0011_eeg_only.vmrk",
    )
    dest = tmp_path / "fixed.vhdr"
    rewrite_header_sidecars(vhdr, dest)
    text = dest.read_text(encoding="utf-8")
    assert "DataFile=BaselineEEG_sub0010_eeg_only.eeg" in text
    assert "MarkerFile=BaselineEEG_sub0010_eeg_only.vmrk" in text


def test_short_recordings_are_rejected() -> None:
    from bcgstudy.simulator_experiment import is_usable_duration

    assert is_usable_duration(8.0, minimum_seconds=60.0) is False
    assert is_usable_duration(610.0, minimum_seconds=60.0) is True
