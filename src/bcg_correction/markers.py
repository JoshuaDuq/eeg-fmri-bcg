"""Marker timing operations."""

import numpy as np
import numpy.typing as npt


class MarkerTimingError(ValueError):
    """Raised when marker timing is invalid or ambiguous."""


def split_volume_blocks(
    volume_samples: npt.ArrayLike,
    *,
    samples_per_volume: int,
    declared_block_starts: npt.ArrayLike | None = None,
) -> tuple[npt.NDArray[np.int64], ...]:
    """Split zero-based volume samples at explicitly declared discontinuities."""
    samples = np.asarray(volume_samples)
    _validate_volume_samples(samples, samples_per_volume)

    boundary_indices = np.flatnonzero(np.diff(samples) != samples_per_volume) + 1
    observed_starts = samples[np.concatenate(([0], boundary_indices))]
    if declared_block_starts is None:
        if boundary_indices.size:
            message = "volume markers contain an undeclared acquisition gap"
            raise MarkerTimingError(message)
    else:
        declared_starts = np.asarray(declared_block_starts)
        _validate_declared_starts(declared_starts)
        if not np.array_equal(declared_starts, observed_starts):
            message = "block declarations mismatch; a marker gap may be undeclared"
            raise MarkerTimingError(message)
    return tuple(np.split(samples.astype(np.int64, copy=False), boundary_indices))


def map_brainvision_position(input_position: int, *, factor: int) -> int:
    """Map a one-based BrainVision position after integer-factor resampling."""
    if type(input_position) is not int or input_position < 1:
        raise MarkerTimingError("BrainVision positions must be positive integers")
    if type(factor) is not int or factor < 1:
        raise MarkerTimingError("resampling factor must be a positive integer")
    return (input_position - 1) // factor + 1


def _validate_volume_samples(samples: np.ndarray, samples_per_volume: int) -> None:
    if samples.ndim != 1 or samples.size == 0:
        message = "volume samples must be a non-empty one-dimensional array"
        raise MarkerTimingError(message)
    if not np.issubdtype(samples.dtype, np.integer):
        raise MarkerTimingError("volume samples must contain integers")
    if np.any(samples < 0):
        raise MarkerTimingError("volume samples cannot be negative")
    if not isinstance(samples_per_volume, int) or samples_per_volume < 1:
        raise MarkerTimingError("samples_per_volume must be a positive integer")
    if np.any(np.diff(samples) <= 0):
        raise MarkerTimingError("volume samples must be strictly increasing")


def _validate_declared_starts(starts: np.ndarray) -> None:
    if starts.ndim != 1 or starts.size == 0:
        raise MarkerTimingError("declared block starts must be a non-empty array")
    if not np.issubdtype(starts.dtype, np.integer):
        raise MarkerTimingError("declared block starts must contain integers")
    if np.any(starts < 0) or np.any(np.diff(starts) <= 0):
        raise MarkerTimingError("declared block starts must be strictly increasing")
