from pathlib import Path

import mne
import numpy as np
from pybv import write_brainvision

from bcg_correction.brainvision import BrainVisionMarker, write_brainvision_markers
from bcgnet.export import bcgnet_output_vhdr, write_bcgnet_recording


def test_bcgnet_output_name_keeps_fastr_stem() -> None:
    src = Path("/data/sub-0000/BaselineEEG_sub0000_2026-02-09_10h56.55.966_fastr.vhdr")
    out = bcgnet_output_vhdr(Path("/out"), "sub-0000", src)
    assert out.name == "BaselineEEG_sub0000_2026-02-09_10h56.55.966_fastr_bcgnet.vhdr"
    assert out.parent.name == "sub-0000"


def test_write_bcgnet_recording_copies_markers_and_ecg(tmp_path: Path) -> None:
    data = np.zeros((2, 1000), dtype=np.float64)
    data[0] = 1e-6
    data[1, 10] = 2e-3
    write_brainvision(
        data=data,
        sfreq=1000.0,
        ch_names=["Cz", "ECG"],
        fname_base="run_fastr",
        folder_out=tmp_path,
        events=[],
        unit="µV",
    )
    marker_path = tmp_path / "run_fastr.vmrk"
    marker_path.unlink()
    markers = (
        BrainVisionMarker("New Segment", "", 1, 1, 0, "20260826123456123456"),
        BrainVisionMarker("Stimulus", "S  1", 101, 1, 0),
    )
    write_brainvision_markers(marker_path, "run_fastr.eeg", markers)
    source = tmp_path / "run_fastr.vhdr"
    info = mne.create_info(["Cz", "ECG"], 1000.0, ch_types=["eeg", "ecg"])
    cleaned_data = data.copy()
    cleaned_data[0] *= 0.5
    cleaned = mne.io.RawArray(cleaned_data, info, verbose="ERROR")
    output = bcgnet_output_vhdr(tmp_path / "out", "sub-0000", source)

    write_bcgnet_recording(cleaned, source, output, overwrite=False)

    raw = mne.io.read_raw_brainvision(output, preload=True, verbose="ERROR")
    assert raw.ch_names == ["Cz", "ECG"]
    np.testing.assert_allclose(raw.get_data()[0], cleaned_data[0], atol=1e-9)
    np.testing.assert_allclose(raw.get_data()[1], data[1], atol=1e-12)
    from bcg_correction.brainvision_io import read_brainvision_recording

    written = read_brainvision_recording(output)
    assert written.markers == markers


def test_write_bcgnet_recording_restores_source_channel_names(tmp_path: Path) -> None:
    data = np.arange(4 * 50, dtype=np.float64).reshape(4, 50) * 1e-6
    write_brainvision(
        data=data,
        sfreq=1000.0,
        ch_names=["Cz", "ECG", "FPz", "CPz"],
        fname_base="run_fastr",
        folder_out=tmp_path,
        events=[],
        unit="µV",
    )
    marker_path = tmp_path / "run_fastr.vmrk"
    marker_path.unlink()
    write_brainvision_markers(
        marker_path,
        "run_fastr.eeg",
        (BrainVisionMarker("New Segment", "", 1, 1, 0, "20260826123456123456"),),
    )
    source = tmp_path / "run_fastr.vhdr"
    info = mne.create_info(
        ["Cz", "ECG", "Fpz", "CPz"],
        1000.0,
        ch_types=["eeg", "ecg", "eeg", "eeg"],
    )
    cleaned = mne.io.RawArray(data.copy(), info, verbose="ERROR")
    output = bcgnet_output_vhdr(tmp_path / "out", "sub-0000", source)

    write_bcgnet_recording(cleaned, source, output, overwrite=False)

    raw = mne.io.read_raw_brainvision(output, preload=True, verbose="ERROR")
    assert raw.ch_names == ["Cz", "ECG", "FPz", "CPz"]
    np.testing.assert_allclose(raw.get_data(), data, atol=1e-9)
