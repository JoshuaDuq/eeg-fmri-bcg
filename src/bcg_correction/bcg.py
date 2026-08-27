"""Bounded cardiac-artifact correction for FASTR recordings."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from numbers import Integral, Real

import mne
import numpy as np
import numpy.typing as npt

_AAS_NEIGHBOR_POOL_FACTOR = 3
_AAS_TAPER_FRACTION = 0.05


class BcgInputError(ValueError):
    """Raised when BCG correction inputs are invalid or insufficient."""


@dataclass(frozen=True, slots=True)
class BcgCorrectionConfig:
    """Validated settings for one bounded BCG correction."""

    method: str
    window_seconds: tuple[float, float]
    ecg_to_bcg_delay_seconds: float
    aas_neighbor_count: int
    pca_obs_components: int

    def __post_init__(self) -> None:
        if not isinstance(self.method, str) or self.method not in {
            "aas",
            "pca_obs",
        }:
            raise BcgInputError("method must be 'aas' or 'pca_obs'")
        if (
            not isinstance(self.window_seconds, tuple)
            or len(self.window_seconds) != 2
            or not all(
                isinstance(value, Real)
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                for value in self.window_seconds
            )
            or self.window_seconds[0] >= self.window_seconds[1]
        ):
            raise BcgInputError(
                "window_seconds must be a finite increasing pair"
            )
        if (
            isinstance(self.ecg_to_bcg_delay_seconds, bool)
            or not isinstance(self.ecg_to_bcg_delay_seconds, Real)
            or not math.isfinite(float(self.ecg_to_bcg_delay_seconds))
        ):
            raise BcgInputError(
                "ecg_to_bcg_delay_seconds must be finite"
            )
        if (
            isinstance(self.aas_neighbor_count, bool)
            or not isinstance(self.aas_neighbor_count, Integral)
            or self.aas_neighbor_count < 1
        ):
            raise BcgInputError(
                "aas_neighbor_count must be a positive integer"
            )
        if (
            isinstance(self.pca_obs_components, bool)
            or not isinstance(self.pca_obs_components, Integral)
            or self.pca_obs_components < 1
        ):
            raise BcgInputError(
                "pca_obs_components must be a positive integer"
            )


@dataclass(frozen=True, slots=True)
class BcgCorrectionResult:
    """Corrected data and the exact samples changed by the splice."""

    data_volts: npt.NDArray[np.float64]
    corrected_samples: npt.NDArray[np.int64]
    method: str


def rr_gap_spans(
    peak_samples: npt.ArrayLike,
    sampling_rate_hz: float,
    maximum_rr_seconds: float,
) -> tuple[tuple[int, int], ...]:
    """Return full 0-based half-open RR spans that exceed the bound."""
    sampling_rate = _validate_sampling_rate(sampling_rate_hz)
    if (
        isinstance(maximum_rr_seconds, bool)
        or not isinstance(maximum_rr_seconds, Real)
        or not math.isfinite(float(maximum_rr_seconds))
        or float(maximum_rr_seconds) <= 0.0
    ):
        raise BcgInputError("maximum_rr_seconds must be finite and positive")
    peaks = _validate_gap_peak_samples(peak_samples)
    if peaks.size < 2:
        return ()
    max_samples = round(float(maximum_rr_seconds) * sampling_rate)
    if max_samples < 1:
        raise BcgInputError("maximum_rr_seconds must cover at least one sample")
    spans = []
    for first, second in pairwise(peaks):
        if int(second) - int(first) > max_samples:
            spans.append((int(first), int(second)))
    return tuple(spans)


def correct_bcg(
    data_volts: npt.ArrayLike,
    peak_samples: npt.ArrayLike,
    sampling_rate_hz: float,
    *,
    channel_names: Sequence[str],
    eeg_picks: npt.ArrayLike,
    ecg_channel_index: int,
    config: BcgCorrectionConfig,
) -> BcgCorrectionResult:
    """Apply bounded AAS or MNE PCA-OBS around explicit BCG artifact anchors."""
    data = _validate_data(data_volts)
    sampling_rate = _validate_sampling_rate(sampling_rate_hz)
    peaks = _validate_peak_samples(peak_samples, data.shape[1])
    names = _validate_channel_names(channel_names, data.shape[0])
    eeg_indices = _validate_eeg_picks(eeg_picks, data.shape[0])
    ecg_index = _validate_ecg_index(ecg_channel_index, data.shape[0])
    if ecg_index in eeg_indices:
        raise BcgInputError("ecg_channel_index cannot be corrected as EEG")

    window_start, window_stop = _window_samples(
        config.window_seconds,
        sampling_rate,
    )
    if config.method == "aas" and window_stop - window_start < 2:
        raise BcgInputError("AAS window must span at least two samples")
    window_bounds, anchor_samples = _complete_windows(
        peaks,
        sampling_rate,
        config.ecg_to_bcg_delay_seconds,
        window_start,
        window_stop,
        data.shape[1],
    )
    if config.method == "aas":
        corrected_samples = _window_union(
            window_bounds,
            sample_count=data.shape[1],
        )
        corrected = _correct_aas(
            data,
            eeg_indices,
            window_bounds,
            anchor_samples,
            config.aas_neighbor_count,
        )
    else:
        corrected_samples = _window_union(
            window_bounds,
            sample_count=data.shape[1],
        )
        corrected = _correct_pca_obs(
            data,
            names,
            eeg_indices,
            anchor_samples,
            corrected_samples,
            sampling_rate,
            config.pca_obs_components,
        )
    corrected[ecg_index] = data[ecg_index]
    return BcgCorrectionResult(
        data_volts=corrected,
        corrected_samples=corrected_samples,
        method=config.method,
    )


def _validate_data(data_volts: npt.ArrayLike) -> np.ndarray:
    values = np.asarray(data_volts)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise BcgInputError("data_volts must have shape (channels, samples)")
    if np.issubdtype(values.dtype, np.bool_) or not np.issubdtype(
        values.dtype,
        np.number,
    ):
        raise BcgInputError("data_volts must contain finite numeric values")
    values = values.astype(np.float64, copy=True)
    if not np.all(np.isfinite(values)):
        raise BcgInputError("data_volts must contain finite numeric values")
    return values


def _validate_sampling_rate(sampling_rate_hz: float) -> float:
    if (
        isinstance(sampling_rate_hz, bool)
        or not isinstance(sampling_rate_hz, Real)
        or not math.isfinite(float(sampling_rate_hz))
        or sampling_rate_hz <= 0.0
    ):
        raise BcgInputError("sampling_rate_hz must be finite and positive")
    return float(sampling_rate_hz)


def _validate_gap_peak_samples(peak_samples: npt.ArrayLike) -> np.ndarray:
    values = np.asarray(peak_samples)
    if values.ndim != 1:
        raise BcgInputError("peak_samples must contain integer samples")
    if values.size == 0:
        return np.empty(0, dtype=np.int64)
    if np.issubdtype(values.dtype, np.bool_) or not np.issubdtype(
        values.dtype,
        np.integer,
    ):
        raise BcgInputError("peak_samples must contain integer samples")
    values = values.astype(np.int64, copy=False)
    if np.any(values < 0):
        raise BcgInputError("peak_samples contain positions outside the recording")
    if values.size >= 2 and np.any(np.diff(values) <= 0):
        raise BcgInputError("peak_samples must be strictly increasing")
    return values


def _validate_peak_samples(
    peak_samples: npt.ArrayLike,
    sample_count: int,
) -> np.ndarray:
    values = np.asarray(peak_samples)
    if values.ndim != 1 or values.size < 2:
        raise BcgInputError("peak_samples must contain at least two events")
    if np.issubdtype(values.dtype, np.bool_) or not np.issubdtype(
        values.dtype,
        np.integer,
    ):
        raise BcgInputError("peak_samples must contain integer samples")
    values = values.astype(np.int64, copy=False)
    if np.any(values < 0) or np.any(values >= sample_count):
        raise BcgInputError("peak_samples contain positions outside the recording")
    if np.any(np.diff(values) <= 0):
        raise BcgInputError("peak_samples must be strictly increasing")
    return values


def _validate_channel_names(
    channel_names: Sequence[str],
    channel_count: int,
) -> tuple[str, ...]:
    names = tuple(channel_names)
    if len(names) != channel_count or not all(
        isinstance(name, str) and name for name in names
    ):
        raise BcgInputError(
            "channel_names must contain one nonempty name per channel"
        )
    if len(set(names)) != len(names):
        raise BcgInputError("channel_names must be unique")
    return names


def _validate_eeg_picks(eeg_picks: npt.ArrayLike, channel_count: int) -> np.ndarray:
    values = np.asarray(eeg_picks)
    if values.ndim != 1 or values.size == 0:
        raise BcgInputError("eeg_picks must contain at least one channel")
    if np.issubdtype(values.dtype, np.bool_) or not np.issubdtype(
        values.dtype,
        np.integer,
    ):
        raise BcgInputError("eeg_picks must contain integer channel indices")
    values = values.astype(np.int64, copy=False)
    if np.any(values < 0) or np.any(values >= channel_count):
        raise BcgInputError("eeg_picks contain an invalid channel index")
    if np.unique(values).size != values.size:
        raise BcgInputError("eeg_picks cannot contain duplicates")
    return values


def _validate_ecg_index(ecg_channel_index: int, channel_count: int) -> int:
    if (
        isinstance(ecg_channel_index, bool)
        or not isinstance(ecg_channel_index, Integral)
        or ecg_channel_index < 0
        or ecg_channel_index >= channel_count
    ):
        raise BcgInputError("ecg_channel_index is outside the recording")
    return int(ecg_channel_index)


def _window_samples(
    window_seconds: tuple[float, float],
    sampling_rate: float,
) -> tuple[int, int]:
    return (
        round(window_seconds[0] * sampling_rate),
        round(window_seconds[1] * sampling_rate),
    )


def _complete_windows(
    peak_samples: np.ndarray,
    sampling_rate: float,
    delay_seconds: float,
    window_start: int,
    window_stop: int,
    sample_count: int,
) -> tuple[tuple[tuple[int, int], ...], np.ndarray]:
    """Keep beats whose full BCG window lies inside the recording.

    Edge events are dropped rather than aborting the run. FASTR outputs are
    trimmed to volume markers, so the first and last beats often lack a
    complete window.
    """
    delay_samples = round(delay_seconds * sampling_rate)
    anchors = peak_samples + delay_samples
    bounds = []
    kept = []
    for anchor in anchors:
        start = int(anchor + window_start)
        stop = int(anchor + window_stop)
        if start < 0 or stop > sample_count or start >= stop:
            continue
        bounds.append((start, stop))
        kept.append(int(anchor))
    if len(kept) < 2:
        raise BcgInputError(
            "fewer than two complete BCG windows remain after dropping "
            "edge events"
        )
    kept_anchors = np.asarray(kept, dtype=np.int64)
    if np.any(np.diff(kept_anchors) <= 0):
        raise BcgInputError("artifact anchors must be strictly increasing")
    return tuple(bounds), kept_anchors


def _window_union(
    bounds: tuple[tuple[int, int], ...],
    *,
    sample_count: int,
) -> np.ndarray:
    coverage = np.zeros(sample_count, dtype=bool)
    for start, stop in bounds:
        coverage[start:stop] = True
    return np.flatnonzero(coverage).astype(np.int64, copy=False)


def _correct_aas(
    data: np.ndarray,
    eeg_indices: np.ndarray,
    window_bounds: tuple[tuple[int, int], ...],
    anchors: np.ndarray,
    neighbor_count: int,
) -> np.ndarray:
    """Leave-one-out local AAS with similar-neighbour templates and LS scale."""
    if len(window_bounds) - 1 < neighbor_count:
        raise BcgInputError(
            "AAS requires at least aas_neighbor_count + 1 complete beats"
        )
    eeg = data[eeg_indices]
    sample_count = data.shape[1]
    correction_sum = np.zeros_like(eeg)
    correction_count = np.zeros(sample_count, dtype=np.int64)
    pool_size = min(
        max(_AAS_NEIGHBOR_POOL_FACTOR * neighbor_count, neighbor_count),
        len(window_bounds) - 1,
    )
    for event_index, (start, stop) in enumerate(window_bounds):
        length = stop - start
        rel_start = start - int(anchors[event_index])
        epoch = eeg[:, start:stop]
        neighbor_epochs = _aas_neighbor_epochs(
            eeg,
            epoch,
            window_bounds,
            anchors,
            event_index,
            rel_start=rel_start,
            length=length,
            neighbor_count=neighbor_count,
            pool_size=pool_size,
            sample_count=sample_count,
        )
        template = neighbor_epochs.mean(axis=0)
        fitted = _scale_template(epoch, template)
        fitted *= _cosine_taper(length)
        correction_sum[:, start:stop] += fitted
        correction_count[start:stop] += 1

    corrected = data.copy()
    corrected_eeg = corrected[eeg_indices]
    covered = correction_count > 0
    corrected_eeg[:, covered] -= (
        correction_sum[:, covered] / correction_count[covered]
    )
    corrected[eeg_indices] = corrected_eeg
    return corrected


def _aas_neighbor_epochs(
    eeg: np.ndarray,
    target: np.ndarray,
    window_bounds: tuple[tuple[int, int], ...],
    anchors: np.ndarray,
    event_index: int,
    *,
    rel_start: int,
    length: int,
    neighbor_count: int,
    pool_size: int,
    sample_count: int,
) -> np.ndarray:
    distances = np.abs(anchors - anchors[event_index])
    distances[event_index] = np.iinfo(np.int64).max
    pool = np.argsort(distances, kind="stable")[:pool_size]
    scored: list[tuple[float, np.ndarray]] = []
    for index in pool:
        start = int(anchors[index]) + rel_start
        stop = start + length
        if start < window_bounds[index][0] or stop > window_bounds[index][1]:
            continue
        if start < 0 or stop > sample_count:
            continue
        neighbor = _demean_channels(eeg[:, start:stop])
        scored.append(
            (_epoch_correlation(_demean_channels(target), neighbor), neighbor)
        )
    if len(scored) < neighbor_count:
        raise BcgInputError(
            f"AAS event {event_index} has {len(scored)} compatible neighbors; "
            f"requires {neighbor_count}"
        )
    scored.sort(key=lambda item: item[0], reverse=True)
    selected = [epoch for _, epoch in scored[:neighbor_count]]
    return np.stack(selected, axis=0)


def _epoch_correlation(left: np.ndarray, right: np.ndarray) -> float:
    x = left.ravel()
    y = right.ravel()
    x = x - x.mean()
    y = y - y.mean()
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom == 0.0:
        return 0.0
    return float(np.dot(x, y) / denom)


def _demean_channels(epoch: np.ndarray) -> np.ndarray:
    return epoch - epoch.mean(axis=1, keepdims=True)


def _scale_template(epoch: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Fit a zero-mean template so channel baseline is not subtracted as BCG."""
    epoch_d = _demean_channels(epoch)
    template_d = _demean_channels(template)
    energies = np.sum(template_d**2, axis=1, keepdims=True)
    amplitudes = np.divide(
        np.sum(epoch_d * template_d, axis=1, keepdims=True),
        energies,
        out=np.ones_like(energies),
        where=energies > 0.0,
    )
    return amplitudes * template_d


def _cosine_taper(sample_count: int) -> np.ndarray:
    taper = np.ones(sample_count, dtype=np.float64)
    edge_samples = max(2, round(_AAS_TAPER_FRACTION * sample_count))
    edge_samples = min(edge_samples, sample_count // 2)
    ramp = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, edge_samples))
    taper[:edge_samples] = ramp
    taper[-edge_samples:] = ramp[::-1]
    return taper


def _correct_pca_obs(
    data: np.ndarray,
    channel_names: tuple[str, ...],
    eeg_indices: np.ndarray,
    anchor_samples: np.ndarray,
    corrected_samples: np.ndarray,
    sampling_rate: float,
    n_components: int,
) -> np.ndarray:
    effective_anchors = _effective_pca_obs_anchors(
        anchor_samples,
        data.shape[1],
    )
    if effective_anchors.size < n_components + 1:
        raise BcgInputError(
            "PCA-OBS requires at least n_components + 1 effective beats"
        )
    peak_range = round(np.median(np.diff(effective_anchors)) / 2.0)
    if n_components > 2 * peak_range + 1:
        raise BcgInputError(
            "pca_obs_components exceeds the effective heartbeat window"
        )

    eeg_data = data[eeg_indices]
    eeg_names = [channel_names[int(index)] for index in eeg_indices]
    raw = mne.io.RawArray(
        eeg_data,
        mne.create_info(
            ch_names=eeg_names,
            sfreq=sampling_rate,
            ch_types=["eeg"] * len(eeg_names),
        ),
        verbose="ERROR",
    )
    corrected_raw = mne.preprocessing.apply_pca_obs(
        raw,
        picks=eeg_names,
        qrs_times=effective_anchors.astype(np.float64) / sampling_rate,
        n_components=n_components,
        copy=True,
        verbose="ERROR",
    )
    try:
        corrected_eeg = corrected_raw.get_data()
    finally:
        raw.close()
        corrected_raw.close()

    corrected = data.copy()
    corrected[eeg_indices] = _splice_corrected_windows(
        eeg_data,
        corrected_eeg,
        corrected_samples,
    )
    return corrected


def _splice_corrected_windows(
    original: np.ndarray,
    corrected: np.ndarray,
    corrected_samples: np.ndarray,
) -> np.ndarray:
    """Keep the correction inside the artifact windows and restore DC outside.

    ``apply_pca_obs`` demeans each channel over the whole recording and also
    interpolates between QRS windows. Measuring the offset on the samples that
    must not move, then copying only the window interiors, is what keeps the
    splice from leaving a step at every epoch boundary.
    """
    spliced = original.copy()
    if corrected_samples.size == 0:
        return spliced
    outside = np.ones(original.shape[1], dtype=bool)
    outside[corrected_samples] = False
    if np.any(outside):
        offset = original[:, outside].mean(axis=1, keepdims=True) - corrected[
            :, outside
        ].mean(axis=1, keepdims=True)
        corrected = corrected + offset
    spliced[:, corrected_samples] = corrected[:, corrected_samples]
    return spliced


def _effective_pca_obs_anchors(
    anchor_samples: np.ndarray,
    sample_count: int,
) -> np.ndarray:
    peak_range = round(np.median(np.diff(anchor_samples)) / 2.0)
    effective_count = anchor_samples.size
    while (
        effective_count > 0
        and anchor_samples[effective_count - 1] + peak_range > sample_count
    ):
        effective_count -= 1
    if effective_count < 2:
        raise BcgInputError(
            "PCA-OBS requires at least two effective heartbeat anchors"
        )
    return anchor_samples[:effective_count]
