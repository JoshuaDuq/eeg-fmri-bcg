"""Descriptive, method-independent heartbeat-locked measurements.

These are not estimates of ground-truth artifact or neural preservation.
Blocks prevent cancellation between distant epochs, but their means still
contain neural activity and finite-sample noise. They are not validation folds.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from scipy.signal import welch


@dataclass(frozen=True, slots=True)
class EvaluationSettings:
    block_counts: tuple[int, ...]
    minimum_beats_per_block: int

    def __post_init__(self):
        counts = self.block_counts
        if (
            not counts
            or any(type(count) is not int or count < 2 for count in counts)
            or tuple(sorted(set(counts))) != counts
        ):
            raise ValueError("evaluation.block_counts must be increasing integers >= 2")
        minimum = self.minimum_beats_per_block
        if type(minimum) is not int or minimum < 2:
            raise ValueError(
                "evaluation.minimum_beats_per_block must be an integer >= 2"
            )


def parse_evaluation(values: Mapping) -> EvaluationSettings:
    keys = {"block_counts", "minimum_beats_per_block"}
    if not isinstance(values, Mapping) or set(values) != keys:
        raise ValueError(f"evaluation requires exactly {sorted(keys)}")
    if not isinstance(values["block_counts"], list):
        raise ValueError("evaluation.block_counts must be a list")
    return EvaluationSettings(
        tuple(values["block_counts"]), values["minimum_beats_per_block"]
    )


def local_locked_energy(epochs: np.ndarray, block_count: int) -> np.ndarray:
    """Channel x time squared block means, weighted by the number of beats."""
    if epochs.ndim != 3 or not np.all(np.isfinite(epochs)):
        raise ValueError("epochs must be finite channel x beat x time data")
    if not 1 <= block_count <= epochs.shape[1]:
        raise ValueError("block count exceeds complete beats")
    blocks = np.array_split(epochs, block_count, axis=1)
    return (
        sum(block.shape[1] * np.square(block.mean(axis=1)) for block in blocks)
        / epochs.shape[1]
    )


def local_locked_rms(epochs: np.ndarray, block_count: int) -> np.ndarray:
    """Per-channel RMS; square before aggregating independent block means."""
    return np.sqrt(local_locked_energy(epochs, block_count).mean(axis=-1))


def epoch_spectrum(epochs: np.ndarray, sampling_rate: float):
    """Average periodograms within channels, without joining epoch boundaries."""
    frequency, power = welch(
        epochs, fs=sampling_rate, nperseg=min(epochs.shape[-1], 1024), axis=-1
    )
    return frequency, power.mean(axis=1)


def band_integral(frequency, power, low, high):
    """Integrate density over frequency bins; result has signal-squared units."""
    mask = (frequency >= low) & (frequency <= high)
    if np.count_nonzero(mask) < 2:
        return np.full(power.shape[:-1], np.nan)
    return np.trapezoid(power[..., mask], frequency[mask], axis=-1)


def divide_or_nan(numerator, denominator):
    numerator, denominator = np.broadcast_arrays(numerator, denominator)
    return np.divide(
        numerator,
        denominator,
        out=np.full(numerator.shape, np.nan),
        where=denominator > 0,
    )


def spectral_locked_fraction(locked_power, variable_power):
    """Within-recording energy fraction, summed over the same EEG channels."""
    return divide_or_nan(
        np.sum(locked_power, axis=0),
        np.sum(locked_power + variable_power, axis=0),
    )
