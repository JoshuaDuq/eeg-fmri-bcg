"""Write BCGNet-cleaned EEG as BrainVision, matching FASTR/AAS layout."""

from __future__ import annotations

from pathlib import Path

import mne
import numpy as np

from bcg_correction.brainvision_io import (
    read_brainvision_recording,
    write_brainvision_recording,
)


def bcgnet_output_vhdr(
    output_root: Path, bids_id: str, fastr_vhdr: Path
) -> Path:
    """Return `{stem}_fastr_bcgnet.vhdr` under `output_root/bids_id/`."""
    return output_root / bids_id / f"{fastr_vhdr.stem}_bcgnet.vhdr"


def write_bcgnet_recording(
    cleaned: mne.io.BaseRaw,
    source_vhdr: Path,
    output_vhdr: Path,
    *,
    overwrite: bool = False,
) -> None:
    """Write cleaned samples with the source recording's markers."""
    output_vhdr = Path(output_vhdr)
    if overwrite:
        for suffix in (".vhdr", ".eeg", ".vmrk"):
            sibling = output_vhdr.with_suffix(suffix)
            if sibling.exists():
                sibling.unlink()
    source = read_brainvision_recording(source_vhdr)
    data, names = _align_channels_to_source(cleaned, source_vhdr)
    write_brainvision_recording(
        data=data,
        sampling_rate=float(cleaned.info["sfreq"]),
        channel_names=names,
        output_vhdr=output_vhdr,
        markers=source.markers,
    )


def _align_channels_to_source(
    cleaned: mne.io.BaseRaw, source_vhdr: Path
) -> tuple[np.ndarray, list[str]]:
    """Reorder cleaned samples to the source FASTR names and order."""
    source = mne.io.read_raw_brainvision(
        source_vhdr, preload=False, verbose="ERROR"
    )
    source_names = list(source.ch_names)
    index = {name.lower(): i for i, name in enumerate(cleaned.ch_names)}
    missing = [name for name in source_names if name.lower() not in index]
    if missing:
        raise ValueError(
            f"cleaned recording is missing source channels: {missing}"
        )
    data = cleaned.get_data()
    aligned = np.empty((len(source_names), data.shape[1]), dtype=np.float64)
    for dest, name in enumerate(source_names):
        aligned[dest] = data[index[name.lower()]]
    return aligned, source_names
