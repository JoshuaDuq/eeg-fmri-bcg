"""MRI-simulator recording discovery and BrainVision sidecar repair.

Simulator EEG is the outside-field reference for the paired thermal 8-13 Hz
experiment. Discovery lives here so the experiment script does not parse
folder names. BCG methods are not applied to these recordings.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_FOLDER = re.compile(r"^sub[-_](\d+)$", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class SimulatorRecording:
    bids_id: str
    source_vhdr: Path
    label: str
    run: int | None


def bids_id_from_folder(name: str) -> str:
    match = _FOLDER.match(name.strip())
    if match is None:
        raise ValueError(f"not a subject folder: {name}")
    return f"sub-{int(match.group(1)):04d}"


def is_usable_duration(seconds: float, *, minimum_seconds: float) -> bool:
    return bool(np.isfinite(seconds) and seconds >= minimum_seconds)


def discover_simulator_recordings(
    root: Path,
    *,
    include_from: int = 10,
    include_to: int = 21,
) -> list[SimulatorRecording]:
    """BrainVision files under ``sub_*/eeg``, restricted to the requested range."""
    from bcgstudy.discovery import label_recordings, list_vhdrs

    found: list[SimulatorRecording] = []
    if not root.is_dir():
        return found
    for folder in sorted(root.iterdir()):
        if not folder.is_dir() or folder.name.startswith("._"):
            continue
        try:
            bids_id = bids_id_from_folder(folder.name)
        except ValueError:
            continue
        number = int(bids_id.split("-")[1])
        if not include_from <= number <= include_to:
            continue
        eeg_dir = folder / "eeg" if (folder / "eeg").is_dir() else folder
        for recording in label_recordings(list_vhdrs(eeg_dir)):
            found.append(
                SimulatorRecording(
                    bids_id=bids_id,
                    source_vhdr=recording.path,
                    label=recording.label,
                    run=recording.run,
                )
            )
    return found


def rewrite_header_sidecars(source: Path, destination: Path) -> None:
    """Point DataFile/MarkerFile at files that share the VHDR stem.

    A few simulator headers were saved with another subject's filenames.
    The .eeg/.vmrk beside the header are the recording; only the pointers
    are wrong.
    """
    text = source.read_text(encoding="utf-8", errors="replace")
    stem = source.stem
    rewritten = []
    for line in text.splitlines():
        if line.startswith("DataFile="):
            rewritten.append(f"DataFile={stem}.eeg")
        elif line.startswith("MarkerFile="):
            rewritten.append(f"MarkerFile={stem}.vmrk")
        else:
            rewritten.append(line)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(rewritten) + "\n", encoding="utf-8")
