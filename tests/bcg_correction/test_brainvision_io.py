from pathlib import Path

import mne
import numpy as np
import pytest
from pybv import write_brainvision

from bcg_correction.brainvision import (
    BrainVisionMarker,
    write_brainvision_markers,
)
from bcg_correction.brainvision_io import (
    BrainVisionInputError,
    read_brainvision_recording,
    select_marker_samples,
    write_brainvision_recording,
)


def make_markers() -> tuple[BrainVisionMarker, ...]:
    return (
        BrainVisionMarker(
            "New Segment",
            "",
            1,
            1,
            0,
            "20260826123456123456",
        ),
        BrainVisionMarker("Volume", "volume-start", 1, 1, 0),
        BrainVisionMarker("Comment", "comma, preserved", 501, 10, 2),
        BrainVisionMarker("Volume", "volume-start", 901, 1, 0),
    )


def make_source_recording(tmp_path: Path) -> Path:
    data = np.zeros((2, 1_000), dtype=np.float64)
    write_brainvision(
        data=data,
        sfreq=1_000.0,
        ch_names=["EEG 001", "ECG"],
        fname_base="source",
        folder_out=tmp_path,
        events=[],
        unit="µV",
    )
    marker_path = tmp_path / "source.vmrk"
    marker_path.unlink()
    write_brainvision_markers(marker_path, "source.eeg", make_markers())
    return tmp_path / "source.vhdr"


def test_read_recording_resolves_header_references_and_markers(tmp_path: Path) -> None:
    vhdr_path = make_source_recording(tmp_path)

    recording = read_brainvision_recording(vhdr_path)

    assert recording.data_path == tmp_path / "source.eeg"
    assert recording.marker_path == tmp_path / "source.vmrk"
    assert recording.markers == make_markers()


def test_read_recording_accepts_recorder_header_identifier(tmp_path: Path) -> None:
    vhdr_path = make_source_recording(tmp_path)
    header = vhdr_path.read_text(encoding="utf-8")
    vhdr_path.write_text(
        header.replace(
            "Brain Vision Data Exchange Header File Version 1.0",
            "BrainVision Data Exchange Header File Version 1.0",
            1,
        ),
        encoding="utf-8",
    )

    recording = read_brainvision_recording(vhdr_path)

    assert recording.data_path == tmp_path / "source.eeg"


def test_read_recording_accepts_version_two_header_identifier(tmp_path: Path) -> None:
    vhdr_path = make_source_recording(tmp_path)
    header = vhdr_path.read_text(encoding="utf-8")
    vhdr_path.write_text(
        header.replace(
            "Brain Vision Data Exchange Header File Version 1.0",
            "Brain Vision Data Exchange Header File Version 2.0",
            1,
        ),
        encoding="utf-8",
    )

    recording = read_brainvision_recording(vhdr_path)

    assert recording.data_path == tmp_path / "source.eeg"


def test_select_marker_samples_requires_exact_configured_match() -> None:
    markers = make_markers()

    samples = select_marker_samples(
        markers,
        marker_type="Volume",
        marker_description="volume-start",
        sample_count=1_000,
    )

    np.testing.assert_array_equal(samples, np.array([0, 900], dtype=np.int64))


def test_select_marker_samples_rejects_missing_or_duplicate_positions() -> None:
    markers = (*make_markers(),
        BrainVisionMarker("Volume", "volume-start", 1, 1, 0),
    )

    with pytest.raises(BrainVisionInputError, match="duplicate"):
        select_marker_samples(
            markers,
            marker_type="Volume",
            marker_description="volume-start",
            sample_count=1_000,
        )

    with pytest.raises(BrainVisionInputError, match="no markers"):
        select_marker_samples(
            markers,
            marker_type="Volume",
            marker_description="missing",
            sample_count=1_000,
        )


def test_write_recording_refuses_existing_output_stem(tmp_path: Path) -> None:
    output = tmp_path / "result.vhdr"
    output.write_text("occupied", encoding="utf-8")

    with pytest.raises(FileExistsError):
        write_brainvision_recording(
            data=np.zeros((1, 10)),
            sampling_rate=100.0,
            channel_names=["EEG 001"],
            output_vhdr=output,
            markers=(),
        )


def test_float_output_stores_microvolts_without_a_scale_factor(
    tmp_path: Path,
) -> None:
    """Analyzer writes float32 with no resolution, so the stored value is the value.

    A reader that ignores ``Resolution`` on a floating-point format, which is a
    reasonable assumption because float needs no scaling, would otherwise read
    every one of our files off by the resolution factor.
    """
    data = np.array([[1.0e-6, -2.5e-6, 0.0], [3.0e-6, 4.0e-6, -1.0e-6]])
    output = tmp_path / "scaled.vhdr"
    write_brainvision_recording(
        data=data,
        sampling_rate=1000.0,
        channel_names=["EEG 001", "EEG 002"],
        output_vhdr=output,
        markers=(),
    )

    header = output.read_text(encoding="utf-8")
    assert "BinaryFormat=IEEE_FLOAT_32" in header
    assert "Ch1=EEG 001,,1,µV" in header

    stored = np.fromfile(output.with_suffix(".eeg"), dtype="<f4").reshape(-1, 2).T
    assert np.allclose(stored, data * 1e6, atol=1e-4)

    raw = mne.io.read_raw_brainvision(output, preload=True, verbose="ERROR")
    assert np.allclose(raw.get_data(), data, atol=1e-12)
