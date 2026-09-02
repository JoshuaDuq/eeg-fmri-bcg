from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from bcg_correction.brainvision import (
    BrainVisionMarker,
    BrainVisionMarkerError,
    read_brainvision_markers,
    write_brainvision_markers,
)


def test_brainvision_marker_preserves_all_fields_and_is_immutable() -> None:
    marker = BrainVisionMarker(
        marker_type="New Segment",
        description="",
        position=1,
        size=1,
        channel=0,
        date="20260826123456123456",
    )

    assert marker == BrainVisionMarker(
        marker_type="New Segment",
        description="",
        position=1,
        size=1,
        channel=0,
        date="20260826123456123456",
    )
    with pytest.raises(FrozenInstanceError):
        marker.position = 2


def test_brainvision_marker_preserves_zero_size() -> None:
    marker = BrainVisionMarker("Comment", "instantaneous", 10, 0, 0)

    assert marker.size == 0


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"marker_type": ""}, "marker_type"),
        ({"marker_type": "Volume\\1Slice"}, "marker_type"),
        ({"description": "line\nbreak"}, "description"),
        ({"position": 0}, "position"),
        ({"position": True}, "position"),
        ({"size": -1}, "size"),
        ({"channel": -1}, "channel"),
        ({"date": "20260230123456123456"}, "date"),
        ({"date": "20260826123456"}, "date"),
    ],
)
def test_brainvision_marker_rejects_invalid_fields(
    changes: dict[str, object],
    message: str,
) -> None:
    values = {
        "marker_type": "Volume",
        "description": "V  1",
        "position": 101,
        "size": 1,
        "channel": 0,
        "date": None,
    }

    with pytest.raises(BrainVisionMarkerError, match=message):
        BrainVisionMarker(**(values | changes))


def test_read_brainvision_markers_preserves_fields_in_mk_order(
    tmp_path: Path,
) -> None:
    marker_path = tmp_path / "recording.vmrk"
    marker_path.write_text(
        "BrainVision Data Exchange Marker File Version 1.0\n"
        "; source fixture\n"
        "[Common Infos]\n"
        "Codepage=UTF-8\n"
        "DataFile=recording.eeg\n"
        "[Marker Infos]\n"
        "Mk1=New Segment,,1,1,0,20260826123456123456\n"
        "Mk2=SyncStatus,Sync On,1,1,0\n"
        "Mk3=Volume,V  1,9001,1,0\n"
        "Mk4=Custom\\1Type,comma\\1description,10000,0,3\n",
        encoding="utf-8",
    )

    data_file_name, markers = read_brainvision_markers(marker_path)

    assert data_file_name == "recording.eeg"
    assert markers == (
        BrainVisionMarker(
            "New Segment",
            "",
            1,
            1,
            0,
            "20260826123456123456",
        ),
        BrainVisionMarker("SyncStatus", "Sync On", 1, 1, 0),
        BrainVisionMarker("Volume", "V  1", 9001, 1, 0),
        BrainVisionMarker("Custom,Type", "comma,description", 10000, 0, 3),
    )


def test_read_brainvision_markers_accepts_version_two_files(
    tmp_path: Path,
) -> None:
    marker_path = tmp_path / "recording.vmrk"
    marker_path.write_text(
        "BrainVision Data Exchange Marker File Version 2.0\n"
        "[Common Infos]\n"
        "Codepage=UTF-8\n"
        "DataFile=recording.eeg\n"
        "[Marker Infos]\n"
        "Mk1=Pulse Artifact,R,101,1,32\n",
        encoding="utf-8",
    )

    data_file_name, markers = read_brainvision_markers(marker_path)

    assert data_file_name == "recording.eeg"
    assert markers == (BrainVisionMarker("Pulse Artifact", "R", 101, 1, 32),)


def test_brainvision_marker_user_infos_round_trip(tmp_path: Path) -> None:
    marker_path = tmp_path / "recording.vmrk"
    original = (
        BrainVisionMarker(
            "Pulse Artifact",
            "R",
            101,
            1,
            32,
            user_infos=(("bool", "BrainVision.CustomMarker", "true"),),
        ),
    )

    write_brainvision_markers(marker_path, "recording.eeg", original)

    assert read_brainvision_markers(marker_path) == ("recording.eeg", original)


def test_read_brainvision_markers_rejects_out_of_order_indices(
    tmp_path: Path,
) -> None:
    marker_path = tmp_path / "recording.vmrk"
    marker_path.write_text(
        "BrainVision Data Exchange Marker File Version 1.0\n"
        "[Common Infos]\n"
        "Codepage=UTF-8\n"
        "DataFile=recording.eeg\n"
        "[Marker Infos]\n"
        "Mk2=SyncStatus,Sync On,2,1,0\n"
        "Mk1=Volume,V  1,1,1,0\n",
        encoding="utf-8",
    )

    with pytest.raises(BrainVisionMarkerError, match="order"):
        read_brainvision_markers(marker_path)


@pytest.mark.parametrize(
    "marker_lines",
    [
        "Mk1=Volume,V  1,1,1,0\nMk1=SyncStatus,Sync On,2,1,0",
        "Mk1=Volume,V  1,1,1,0\nMk3=SyncStatus,Sync On,2,1,0",
        "Mk0=Volume,V  1,1,1,0",
    ],
)
def test_read_brainvision_markers_rejects_noncontiguous_indices(
    tmp_path: Path,
    marker_lines: str,
) -> None:
    marker_path = tmp_path / "recording.vmrk"
    marker_path.write_text(
        "BrainVision Data Exchange Marker File Version 1.0\n"
        "[Common Infos]\n"
        "Codepage=UTF-8\n"
        "DataFile=recording.eeg\n"
        "[Marker Infos]\n"
        f"{marker_lines}\n",
        encoding="utf-8",
    )

    with pytest.raises(BrainVisionMarkerError, match="indices"):
        read_brainvision_markers(marker_path)


@pytest.mark.parametrize(
    "body",
    [
        (
            "[Common Infos]\nCodepage=UTF-8\nDataFile=recording.eeg\n"
            "[Common Infos]\n[Marker Infos]"
        ),
        (
            "[Common Infos]\nCodepage=UTF-8\nDataFile=one.eeg\n"
            "DataFile=two.eeg\n[Marker Infos]"
        ),
        (
            "[Common Infos]\nCodepage=UTF-8\nDataFile=recording.eeg\n"
            "[Marker Infos]\n[Marker Infos]"
        ),
        "[Common Infos]\nDataFile=recording.eeg\n[Marker Infos]",
        (
            "[Common Infos]\nCodepage=ANSI\nDataFile=recording.eeg\n"
            "[Marker Infos]"
        ),
        (
            "[Common Infos]\nCodepage=UTF-8\nDataFile=recording.eeg\n"
            "Extra=unsupported\n[Marker Infos]"
        ),
        (
            "[Common Infos]\nCodepage=UTF-8\nDataFile=recording.eeg\n"
            "[Marker Infos]\n[Comment]"
        ),
    ],
)
def test_read_brainvision_markers_rejects_ambiguous_structure(
    tmp_path: Path,
    body: str,
) -> None:
    marker_path = tmp_path / "recording.vmrk"
    marker_path.write_text(
        "BrainVision Data Exchange Marker File Version 1.0\n" f"{body}\n",
        encoding="utf-8",
    )

    with pytest.raises(BrainVisionMarkerError):
        read_brainvision_markers(marker_path)


@pytest.mark.parametrize(
    "data_file_name",
    [
        "/absolute/recording.eeg",
        "../recording.eeg",
        "subdirectory/recording.eeg",
        r"subdirectory\recording.eeg",
        ".",
        "..",
    ],
)
def test_brainvision_marker_io_rejects_nonlocal_data_file_names(
    tmp_path: Path,
    data_file_name: str,
) -> None:
    source_path = tmp_path / "source.vmrk"
    source_path.write_text(
        "BrainVision Data Exchange Marker File Version 1.0\n"
        "[Common Infos]\n"
        "Codepage=UTF-8\n"
        f"DataFile={data_file_name}\n"
        "[Marker Infos]\n",
        encoding="utf-8",
    )

    with pytest.raises(BrainVisionMarkerError, match="same-directory"):
        read_brainvision_markers(source_path)
    with pytest.raises(BrainVisionMarkerError, match="same-directory"):
        write_brainvision_markers(tmp_path / "output.vmrk", data_file_name, ())


@pytest.mark.parametrize(
    "marker_line",
    [
        "Mk01=Volume,V  1,1,1,0",
        "Mk1=Volume,V  1, 1,1,0",
        "Mk1=Volume,V  1,1,1",
        "Mk1 Volume,V  1,1,1,0",
        "Marker1=Volume,V  1,1,1,0",
        "Mk1=Volume,V  1,1.0,1,0",
        "Mk1=Volume,V  1,0,1,0",
        "Mk1=Volume,V  1,1,-1,0",
        "Mk1=Volume,V  1,1,1,-1",
        "Mk1=New Segment,,1,1,0,20260230123456123456",
    ],
)
def test_read_brainvision_markers_rejects_malformed_marker_lines(
    tmp_path: Path,
    marker_line: str,
) -> None:
    marker_path = tmp_path / "recording.vmrk"
    marker_path.write_text(
        "BrainVision Data Exchange Marker File Version 1.0\n"
        "[Common Infos]\n"
        "Codepage=UTF-8\n"
        "DataFile=recording.eeg\n"
        "[Marker Infos]\n"
        f"{marker_line}\n",
        encoding="utf-8",
    )

    with pytest.raises(BrainVisionMarkerError):
        read_brainvision_markers(marker_path)


def test_write_brainvision_markers_emits_standard_lossless_file(
    tmp_path: Path,
) -> None:
    marker_path = tmp_path / "corrected.vmrk"
    markers = (
        BrainVisionMarker(
            "New Segment",
            "",
            1,
            1,
            0,
            "20260826123456123456",
        ),
        BrainVisionMarker("SyncStatus", "Sync On", 1, 1, 0),
        BrainVisionMarker("Volume", "V  1", 9001, 1, 0),
        BrainVisionMarker("Custom,Type", "comma,description", 10000, 0, 3),
    )

    write_brainvision_markers(marker_path, "corrected.eeg", markers)

    assert marker_path.read_text(encoding="utf-8") == (
        "BrainVision Data Exchange Marker File Version 1.0\n"
        "[Common Infos]\n"
        "Codepage=UTF-8\n"
        "DataFile=corrected.eeg\n"
        "\n"
        "[Marker Infos]\n"
        "Mk1=New Segment,,1,1,0,20260826123456123456\n"
        "Mk2=SyncStatus,Sync On,1,1,0\n"
        "Mk3=Volume,V  1,9001,1,0\n"
        "Mk4=Custom\\1Type,comma\\1description,10000,0,3\n"
    )


def test_brainvision_markers_are_equal_after_read_write_read(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.vmrk"
    output_path = tmp_path / "output.vmrk"
    source_path.write_text(
        "BrainVision Data Exchange Marker File Version 1.0\n"
        "[Common Infos]\n"
        "Codepage=UTF-8\n"
        "DataFile=recording.eeg\n"
        "[Marker Infos]\n"
        "Mk1=New Segment,,1,1,0,20260826123456123456\n"
        "Mk2=Volume,V  1,9001,1,0\n"
        "Mk3=SyncStatus,Sync\\1On,9500,0,7\n",
        encoding="utf-8",
    )

    original = read_brainvision_markers(source_path)
    write_brainvision_markers(output_path, *original)

    assert read_brainvision_markers(output_path) == original


def test_write_brainvision_markers_refuses_to_overwrite(tmp_path: Path) -> None:
    marker_path = tmp_path / "existing.vmrk"
    marker_path.write_text("existing content", encoding="utf-8")

    with pytest.raises(FileExistsError):
        write_brainvision_markers(marker_path, "recording.eeg", ())

    assert marker_path.read_text(encoding="utf-8") == "existing content"
