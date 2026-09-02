"""Strict parsing for bounded-correction provenance sidecars."""

from __future__ import annotations

import json
from dataclasses import dataclass
from math import isfinite
from pathlib import Path

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, slots=True)
class CorrectionProvenance:
    """Scoring context recorded by a bounded correction."""

    peak_samples: npt.NDArray[np.int64]
    delay_seconds: float
    window_seconds: tuple[float, float]
    gap_fraction: float


def _finite_float(value: object, *, name: str, path: Path) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric: {path}")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be finite: {path}")
    return result


def load_correction_provenance(vhdr: Path) -> CorrectionProvenance | None:
    path = vhdr.parent / f"{vhdr.stem}.bcg.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"provenance must be a mapping: {path}")
    required = {
        "peak_samples",
        "ecg_to_bcg_delay_seconds",
        "window_seconds",
        "rr_gap_fraction",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError(f"provenance is missing {', '.join(missing)}: {path}")

    peaks = payload["peak_samples"]
    if (
        not isinstance(peaks, list)
        or not peaks
        or any(not isinstance(value, int) or isinstance(value, bool) for value in peaks)
    ):
        raise ValueError(f"peak_samples must be a non-empty integer list: {path}")
    peak_samples = np.asarray(peaks, dtype=np.int64)
    if np.any(peak_samples < 0) or np.any(np.diff(peak_samples) <= 0):
        raise ValueError(f"peak_samples must be non-negative and increasing: {path}")

    delay = _finite_float(
        payload["ecg_to_bcg_delay_seconds"],
        name="ecg_to_bcg_delay_seconds",
        path=path,
    )
    window = payload["window_seconds"]
    if not isinstance(window, list) or len(window) != 2:
        raise ValueError(f"window_seconds must contain two values: {path}")
    window_seconds = (
        _finite_float(window[0], name="window_seconds", path=path),
        _finite_float(window[1], name="window_seconds", path=path),
    )
    if window_seconds[0] >= window_seconds[1]:
        raise ValueError(f"window_seconds must be increasing: {path}")
    gap_fraction = _finite_float(
        payload["rr_gap_fraction"], name="rr_gap_fraction", path=path
    )
    if not 0.0 <= gap_fraction <= 1.0:
        raise ValueError(f"rr_gap_fraction must be between 0 and 1: {path}")
    return CorrectionProvenance(
        peak_samples=peak_samples,
        delay_seconds=delay,
        window_seconds=window_seconds,
        gap_fraction=gap_fraction,
    )
