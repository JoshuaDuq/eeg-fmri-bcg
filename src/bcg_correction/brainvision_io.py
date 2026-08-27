"""Strict BrainVision input discovery, marker selection, and output writing."""

from __future__ import annotations

import math
import tempfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path

import numpy as np
import numpy.typing as npt
from pybv import write_brainvision

from .brainvision import (
    BrainVisionMarker,
    read_brainvision_markers,
    write_brainvision_markers,
)
from .markers import map_brainvision_position
from .window import OutputWindow


class BrainVisionInputError(ValueError):
    """Raised when a BrainVision recording is incomplete or ambiguous."""


_HEADER_IDENTIFIERS = frozenset(
    {
        "Brain Vision Data Exchange Header File Version 1.0",
        "BrainVision Data Exchange Header File Version 1.0",
        "Brain Vision Data Exchange Header File Version 2.0",
        "BrainVision Data Exchange Header File Version 2.0",
    }
)


@dataclass(frozen=True, slots=True)
class BrainVisionRecording:
    """Resolved BrainVision files and their losslessly parsed markers."""

    header_path: Path
    data_path: Path
    marker_path: Path
    markers: tuple[BrainVisionMarker, ...]


def read_brainvision_recording(path: str | Path) -> BrainVisionRecording:
    """Resolve a BrainVision header and read its marker file strictly."""
    header_path = Path(path).expanduser().resolve()
    if header_path.suffix.lower() != ".vhdr":
        raise BrainVisionInputError("the BrainVision input must have a .vhdr suffix")
    if not header_path.is_file():
        raise BrainVisionInputError(f"BrainVision header does not exist: {header_path}")

    data_file_name, marker_file_name = _read_header_references(header_path)
    data_path = _resolve_local_reference(header_path, data_file_name, "DataFile")
    marker_path = _resolve_local_reference(
        header_path,
        marker_file_name,
        "MarkerFile",
    )
    marker_data_file_name, markers = read_brainvision_markers(marker_path)
    if marker_data_file_name != data_file_name:
        raise BrainVisionInputError(
            "BrainVision header and marker file reference different data files"
        )
    if not data_path.is_file():
        raise BrainVisionInputError(f"BrainVision data does not exist: {data_path}")
    return BrainVisionRecording(
        header_path=header_path,
        data_path=data_path,
        marker_path=marker_path,
        markers=markers,
    )


def select_marker_samples(
    markers: Sequence[BrainVisionMarker],
    *,
    marker_type: str,
    marker_description: str,
    sample_count: int,
) -> np.ndarray:
    """Select exact configured markers and return zero-based sample positions."""
    if not isinstance(marker_type, str) or not marker_type:
        raise BrainVisionInputError("marker_type must be a nonempty string")
    if not isinstance(marker_description, str) or not marker_description:
        raise BrainVisionInputError(
            "marker_description must be a nonempty string"
        )
    if isinstance(sample_count, bool) or not isinstance(sample_count, Integral):
        raise BrainVisionInputError("sample_count must be a positive integer")
    sample_count = int(sample_count)
    if sample_count < 1:
        raise BrainVisionInputError("sample_count must be a positive integer")

    selected = tuple(
        marker
        for marker in markers
        if marker.marker_type == marker_type
        and marker.description == marker_description
    )
    if not selected:
        raise BrainVisionInputError(
            f"no markers match type {marker_type!r} and description "
            f"{marker_description!r}"
        )
    positions = np.asarray(
        [marker.position - 1 for marker in selected],
        dtype=np.int64,
    )
    if np.any(positions >= sample_count):
        raise BrainVisionInputError(
            "configured markers contain positions outside the recording"
        )
    if np.unique(positions).size != positions.size:
        raise BrainVisionInputError("configured markers contain duplicate positions")
    if np.any(np.diff(positions) <= 0):
        raise BrainVisionInputError(
            "configured markers must be in strictly increasing file order"
        )
    return positions


def resample_markers(
    markers: Iterable[BrainVisionMarker],
    *,
    factor: int,
    window: OutputWindow | None = None,
) -> tuple[BrainVisionMarker, ...]:
    """Map marker positions through an output window and integer decimation.

    Marker positions are one-based; ``window`` bounds are zero-based and
    half-open. Markers outside the window are dropped rather than clamped, so a
    corrected recording never claims an event it does not contain.
    """
    if isinstance(factor, bool) or not isinstance(factor, int) or factor < 1:
        raise BrainVisionInputError("resampling factor must be a positive integer")
    transformed = []
    for marker in markers:
        position = marker.position
        if window is not None:
            index = position - 1
            if index < window.start or index >= window.stop:
                continue
            position = index - window.start + 1
        transformed.append(
            BrainVisionMarker(
                marker_type=marker.marker_type,
                description=marker.description,
                position=map_brainvision_position(position, factor=factor),
                size=(marker.size + factor - 1) // factor,
                channel=marker.channel,
                date=marker.date,
                user_infos=marker.user_infos,
            )
        )
    return tuple(transformed)


def write_brainvision_recording(
    *,
    data: npt.NDArray[np.floating],
    sampling_rate: float,
    channel_names: Sequence[str],
    output_vhdr: str | Path,
    markers: Iterable[BrainVisionMarker],
) -> None:
    """Write a complete BrainVision recording without overwriting output files.

    ``data`` is expected in SI volts, matching MNE's raw-data convention. The
    binary output is written as float32 with BrainVision's conventional microvolt
    unit declaration.
    """
    recording = np.asarray(data)
    _validate_output_data(recording, sampling_rate, channel_names)
    marker_values = tuple(markers)
    if any(not isinstance(marker, BrainVisionMarker) for marker in marker_values):
        raise BrainVisionInputError(
            "markers must contain BrainVisionMarker instances"
        )

    header_path = Path(output_vhdr).expanduser().resolve()
    if header_path.suffix.lower() != ".vhdr":
        raise BrainVisionInputError("output_vhdr must have a .vhdr suffix")
    output_paths = tuple(
        header_path.with_suffix(suffix)
        for suffix in (".eeg", ".vmrk", ".vhdr")
    )
    if any(path.exists() for path in output_paths):
        raise FileExistsError(
            f"one or more BrainVision output files already exist for {header_path}"
        )
    header_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        dir=header_path.parent,
        prefix=f".{header_path.stem}-",
    ) as temporary_directory:
        temporary_directory_path = Path(temporary_directory)
        write_brainvision(
            data=recording,
            sfreq=float(sampling_rate),
            ch_names=list(channel_names),
            fname_base=header_path.stem,
            folder_out=temporary_directory_path,
            events=[],
            unit="µV",
            fmt="binary_float32",
            resolution=1.0,
            overwrite=False,
        )
        temporary_marker_path = temporary_directory_path / f"{header_path.stem}.vmrk"
        temporary_marker_path.unlink()
        write_brainvision_markers(
            temporary_marker_path,
            f"{header_path.stem}.eeg",
            marker_values,
        )
        for suffix in (".eeg", ".vmrk", ".vhdr"):
            temporary_path = temporary_directory_path / f"{header_path.stem}{suffix}"
            temporary_path.rename(header_path.with_suffix(suffix))


def _read_header_references(path: Path) -> tuple[str, str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0] not in _HEADER_IDENTIFIERS:
        raise BrainVisionInputError("invalid BrainVision header identifier")

    in_common_infos = False
    references: dict[str, str] = {}
    for line in lines[1:]:
        if not line or line.startswith(";"):
            continue
        if line.startswith("["):
            if line == "[Common Infos]":
                in_common_infos = True
                continue
            if in_common_infos:
                break
        if not in_common_infos:
            continue
        key, separator, value = line.partition("=")
        if separator and key in ("DataFile", "MarkerFile"):
            if key in references:
                raise BrainVisionInputError(f"duplicate BrainVision {key} declaration")
            references[key] = value

    missing = [key for key in ("DataFile", "MarkerFile") if key not in references]
    if missing:
        raise BrainVisionInputError(
            f"BrainVision header is missing {', '.join(missing)}"
        )
    return references["DataFile"], references["MarkerFile"]


def _resolve_local_reference(
    header_path: Path,
    reference: str,
    field_name: str,
) -> Path:
    if not reference or reference in (".", "..") or any(
        separator in reference for separator in ("/", "\\")
    ):
        raise BrainVisionInputError(
            f"BrainVision {field_name} must be a same-directory filename"
        )
    return header_path.parent / reference


def _validate_output_data(
    data: np.ndarray,
    sampling_rate: float,
    channel_names: Sequence[str],
) -> None:
    if data.ndim != 2 or data.shape[0] == 0 or data.shape[1] == 0:
        raise BrainVisionInputError("data must have shape (channels, samples)")
    if not np.issubdtype(data.dtype, np.number) or not np.all(np.isfinite(data)):
        raise BrainVisionInputError("data must contain finite numeric values")
    if (
        isinstance(sampling_rate, bool)
        or not isinstance(sampling_rate, (int, float))
        or not math.isfinite(float(sampling_rate))
        or sampling_rate <= 0.0
    ):
        raise BrainVisionInputError("sampling_rate must be finite and positive")
    if len(channel_names) != data.shape[0] or not all(
        isinstance(name, str) and name for name in channel_names
    ):
        raise BrainVisionInputError(
            "channel_names must contain one nonempty string per channel"
        )
    if len(set(channel_names)) != len(channel_names):
        raise BrainVisionInputError("channel_names must be unique")
