"""Shared detector provenance and descriptive alpha-peak measurement."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import numpy.typing as npt

from bcg_correction.provenance import CorrectionProvenance, load_correction_provenance

ALPHA_PEAK_BAND = (8.0, 13.0)


def alpha_peak_height(
    freqs: npt.NDArray[np.floating],
    pxx: npt.NDArray[np.floating],
    *,
    low: float = ALPHA_PEAK_BAND[0],
    high: float = ALPHA_PEAK_BAND[1],
) -> float | None:
    mask = (freqs >= low) & (freqs <= high)
    if not np.any(mask):
        return None
    return float(np.max(pxx[mask]))


def shared_detector_provenance(
    provenances: Mapping[str, CorrectionProvenance],
) -> CorrectionProvenance | None:
    iterator = iter(provenances.items())
    first = next(iterator, None)
    if first is None:
        return None
    reference_arm, reference = first
    for arm, candidate in iterator:
        matches = (
            np.array_equal(reference.peak_samples, candidate.peak_samples)
            and reference.delay_seconds == candidate.delay_seconds
            and reference.window_seconds == candidate.window_seconds
            and reference.gap_fraction == candidate.gap_fraction
        )
        if not matches:
            raise ValueError(
                f"inconsistent detector provenance between {reference_arm} and {arm}"
            )
    return reference


def load_shared_detector_provenance(
    vhdr_by_arm: Mapping[str, Path],
) -> CorrectionProvenance | None:
    provenances: dict[str, CorrectionProvenance] = {}
    for arm, vhdr in vhdr_by_arm.items():
        provenance = load_correction_provenance(vhdr)
        if provenance is None:
            raise FileNotFoundError(
                f"missing detector provenance for {arm}: "
                f"{vhdr.with_suffix('.bcg.json')}"
            )
        provenances[arm] = provenance
    return shared_detector_provenance(provenances)
