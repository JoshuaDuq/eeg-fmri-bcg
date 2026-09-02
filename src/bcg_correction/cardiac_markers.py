"""Independent cardiac marker output and post-hoc marker-train audits."""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path

import mne
import numpy as np
import numpy.typing as npt

from .brainvision import (
    BrainVisionMarker,
    read_brainvision_markers,
    write_brainvision_markers,
)
from .brainvision_io import read_brainvision_recording

PULSE_MARKER_TYPE = "Pulse Artifact"
PULSE_MARKER_DESCRIPTION = "R"


class CardiacMarkerError(ValueError):
    """Raised when cardiac markers cannot be validated or written."""


@dataclass(frozen=True, slots=True)
class MarkerAudit:
    """One-to-one agreement summary for two cardiac marker trains."""

    analyzer_samples: npt.NDArray[np.int64]
    detected_samples: npt.NDArray[np.int64]
    matched_count: int
    tolerance_samples: int
    median_lag_samples: float | None
    lag_iqr_samples: float | None


@dataclass(frozen=True, slots=True)
class DetectionSummary:
    """Output paths and status for one independent marker derivation."""

    output_vhdr: Path
    provenance_json: Path
    marker_count: int
    status: str


def _validate_marker_samples(
    samples: npt.ArrayLike,
    field_name: str,
) -> np.ndarray:
    values = np.asarray(samples)
    if values.ndim != 1:
        raise CardiacMarkerError(f"{field_name} must be one-dimensional")
    if values.size == 0:
        return np.empty(0, dtype=np.int64)
    if np.issubdtype(values.dtype, np.bool_) or not np.issubdtype(
        values.dtype,
        np.integer,
    ):
        raise CardiacMarkerError(f"{field_name} must contain integer samples")
    values = values.astype(np.int64, copy=False)
    if np.any(values < 0):
        raise CardiacMarkerError(f"{field_name} cannot contain negative samples")
    if np.unique(values).size != values.size:
        raise CardiacMarkerError(f"{field_name} cannot contain duplicate samples")
    return np.sort(values)


def _validate_peak_samples(
    peak_samples: npt.ArrayLike,
    sample_count: int,
) -> np.ndarray:
    if isinstance(sample_count, bool) or not isinstance(sample_count, Integral):
        raise CardiacMarkerError("sample_count must be a positive integer")
    if sample_count < 1:
        raise CardiacMarkerError("sample_count must be a positive integer")
    samples = _validate_marker_samples(peak_samples, "peak_samples")
    if np.any(samples >= sample_count):
        raise CardiacMarkerError(
            "peak_samples contain positions outside the recording"
        )
    return samples


def _validate_tolerance(tolerance_samples: int) -> int:
    if (
        isinstance(tolerance_samples, bool)
        or not isinstance(tolerance_samples, Integral)
        or tolerance_samples < 0
    ):
        raise CardiacMarkerError(
            "tolerance_samples must be a nonnegative integer"
        )
    return int(tolerance_samples)


def _alignment_is_better(
    candidate_count: int,
    candidate_cost: float,
    current_count: int,
    current_cost: float,
) -> bool:
    return candidate_count > current_count or (
        candidate_count == current_count and candidate_cost < current_cost
    )


def _copy_better_alignment(
    match_counts: np.ndarray,
    lag_costs: np.ndarray,
    decisions: np.ndarray,
    detected_index: int,
    analyzer_index: int,
    previous_detected_index: int,
    previous_analyzer_index: int,
    *,
    decision: int,
) -> None:
    candidate_count = match_counts[previous_detected_index, previous_analyzer_index]
    candidate_cost = lag_costs[previous_detected_index, previous_analyzer_index]
    if _alignment_is_better(
        candidate_count,
        candidate_cost,
        match_counts[detected_index, analyzer_index],
        lag_costs[detected_index, analyzer_index],
    ):
        match_counts[detected_index, analyzer_index] = candidate_count
        lag_costs[detected_index, analyzer_index] = candidate_cost
        decisions[detected_index, analyzer_index] = decision


def _match_marker_lags(
    analyzer: np.ndarray,
    detected: np.ndarray,
    tolerance: int,
) -> list[int]:
    detected_count = detected.size
    analyzer_count = analyzer.size
    match_counts = np.zeros(
        (detected_count + 1, analyzer_count + 1),
        dtype=np.int32,
    )
    lag_costs = np.zeros(
        (detected_count + 1, analyzer_count + 1),
        dtype=np.float64,
    )
    decisions = np.zeros(
        (detected_count + 1, analyzer_count + 1),
        dtype=np.uint8,
    )

    for detected_index in range(1, detected_count + 1):
        for analyzer_index in range(1, analyzer_count + 1):
            _copy_better_alignment(
                match_counts,
                lag_costs,
                decisions,
                detected_index,
                analyzer_index,
                detected_index - 1,
                analyzer_index,
                decision=2,
            )
            _copy_better_alignment(
                match_counts,
                lag_costs,
                decisions,
                detected_index,
                analyzer_index,
                detected_index,
                analyzer_index - 1,
                decision=3,
            )
            lag = int(detected[detected_index - 1] - analyzer[analyzer_index - 1])
            if abs(lag) <= tolerance:
                previous_count = match_counts[detected_index - 1, analyzer_index - 1]
                previous_cost = lag_costs[
                    detected_index - 1,
                    analyzer_index - 1,
                ]
                candidate_count = previous_count + 1
                candidate_cost = previous_cost + abs(lag)
                if _alignment_is_better(
                    candidate_count,
                    candidate_cost,
                    match_counts[detected_index, analyzer_index],
                    lag_costs[detected_index, analyzer_index],
                ):
                    match_counts[detected_index, analyzer_index] = candidate_count
                    lag_costs[detected_index, analyzer_index] = candidate_cost
                    decisions[detected_index, analyzer_index] = 1

    lags: list[int] = []
    detected_index = detected_count
    analyzer_index = analyzer_count
    while detected_index and analyzer_index:
        decision = decisions[detected_index, analyzer_index]
        if decision == 1:
            lags.append(
                int(
                    detected[detected_index - 1]
                    - analyzer[analyzer_index - 1]
                )
            )
            detected_index -= 1
            analyzer_index -= 1
        elif decision == 2:
            detected_index -= 1
        else:
            analyzer_index -= 1
    lags.reverse()
    return lags


def _rewrite_header_references(
    header_text: str,
    *,
    data_file_name: str,
    marker_file_name: str,
) -> str:
    lines = header_text.splitlines(keepends=True)
    rewritten: list[str] = []
    in_common_infos = False
    data_replaced = False
    marker_replaced = False
    for line in lines:
        content = line.rstrip("\r\n")
        line_ending = line[len(content) :]
        if content.startswith("["):
            in_common_infos = content == "[Common Infos]"
        if in_common_infos:
            key, separator, _ = content.partition("=")
            if separator == "=" and key == "DataFile":
                content = f"DataFile={data_file_name}"
                data_replaced = True
            elif separator == "=" and key == "MarkerFile":
                content = f"MarkerFile={marker_file_name}"
                marker_replaced = True
        rewritten.append(content + line_ending)
    if not data_replaced or not marker_replaced:
        raise CardiacMarkerError(
            "BrainVision header references could not be rewritten"
        )
    return "".join(rewritten)


def _brainvision_sample_count(header_path: Path) -> int:
    raw = mne.io.read_raw_brainvision(
        header_path,
        preload=False,
        verbose="ERROR",
    )
    try:
        return int(raw.n_times)
    finally:
        raw.close()


def validate_fastr_marker_input(
    markers: Sequence[BrainVisionMarker],
) -> None:
    if any(not isinstance(marker, BrainVisionMarker) for marker in markers):
        raise CardiacMarkerError(
            "markers must contain BrainVisionMarker instances"
        )
    if any(
        marker.marker_type == PULSE_MARKER_TYPE
        and marker.description == PULSE_MARKER_DESCRIPTION
        for marker in markers
    ):
        raise CardiacMarkerError(
            "FASTR-only input must not contain Pulse Artifact,R markers"
        )


def append_pulse_markers(
    markers: Sequence[BrainVisionMarker],
    peak_samples: npt.ArrayLike,
    *,
    sample_count: int,
) -> tuple[BrainVisionMarker, ...]:
    validate_fastr_marker_input(markers)
    validated_samples = _validate_peak_samples(peak_samples, sample_count)
    preserved_markers = tuple(markers)
    detector_markers = tuple(
        BrainVisionMarker(
            marker_type=PULSE_MARKER_TYPE,
            description=PULSE_MARKER_DESCRIPTION,
            position=int(sample) + 1,
            size=1,
            channel=0,
        )
        for sample in validated_samples
    )
    return preserved_markers + detector_markers


def audit_marker_trains(
    analyzer_samples: npt.ArrayLike,
    detected_samples: npt.ArrayLike,
    *,
    tolerance_samples: int,
) -> MarkerAudit:
    analyzer = _validate_marker_samples(analyzer_samples, "analyzer_samples")
    detected = _validate_marker_samples(detected_samples, "detected_samples")
    tolerance = _validate_tolerance(tolerance_samples)
    matched_lags = _match_marker_lags(analyzer, detected, tolerance)
    if matched_lags:
        lags = np.asarray(matched_lags, dtype=np.float64)
        median_lag = float(np.median(lags))
        lag_iqr = float(np.percentile(lags, 75) - np.percentile(lags, 25))
    else:
        median_lag = None
        lag_iqr = None
    return MarkerAudit(
        analyzer_samples=analyzer,
        detected_samples=detected,
        matched_count=len(matched_lags),
        tolerance_samples=tolerance,
        median_lag_samples=median_lag,
        lag_iqr_samples=lag_iqr,
    )


def write_marker_recording(
    source_vhdr: str | Path,
    output_vhdr: str | Path,
    *,
    peak_samples: npt.ArrayLike,
) -> Path:
    source = read_brainvision_recording(source_vhdr)
    output_header = Path(output_vhdr).expanduser().resolve()
    if output_header.suffix.lower() != ".vhdr":
        raise CardiacMarkerError("output_vhdr must have a .vhdr suffix")
    output_data = output_header.with_suffix(".eeg")
    output_marker = output_header.with_suffix(".vmrk")
    output_paths = (output_header, output_data, output_marker)
    if any(path.exists() for path in output_paths):
        raise FileExistsError(
            f"one or more BrainVision output files already exist for "
            f"{output_header}"
        )

    sample_count = _brainvision_sample_count(source.header_path)
    peaks = _validate_peak_samples(peak_samples, sample_count)
    data_file_name, _ = read_brainvision_markers(source.marker_path)
    header_text = source.header_path.read_text(encoding="utf-8")
    output_data_name = output_data.name
    output_marker_name = output_marker.name
    references_differ = (
        source.header_path.parent != output_header.parent
        or data_file_name != output_data_name
        or source.marker_path.name != output_marker_name
    )
    if references_differ:
        header_text = _rewrite_header_references(
            header_text,
            data_file_name=output_data_name,
            marker_file_name=output_marker_name,
        )

    output_header.parent.mkdir(parents=True, exist_ok=True)
    markers = append_pulse_markers(
        source.markers,
        peaks,
        sample_count=sample_count,
    )
    with tempfile.TemporaryDirectory(
        dir=output_header.parent,
        prefix=f".{output_header.stem}-",
    ) as temporary_directory:
        temporary_path = Path(temporary_directory)
        temporary_header = temporary_path / output_header.name
        temporary_data = temporary_path / output_data.name
        temporary_marker = temporary_path / output_marker.name
        temporary_header.write_text(header_text, encoding="utf-8", newline="")
        shutil.copyfile(source.data_path, temporary_data)
        write_brainvision_markers(
            temporary_marker,
            output_data.name,
            markers,
        )
        temporary_data.rename(output_data)
        temporary_marker.rename(output_marker)
        temporary_header.rename(output_header)
    return output_header
