"""Multi-participant screen of cross-fitted ECG-conditioned BCG templates."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, fields
from itertools import pairwise
from numbers import Integral, Real
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import yaml
from scipy.signal import resample_poly

from bcg_correction.adaptive import (
    AdaptiveEpochConfig,
    ContinuousMetricVariants,
    EpochCorrectionMetrics,
    apply_template_predictions_to_recording,
    combine_feature_groups,
    contiguous_cross_fit_training_mask,
    continuous_epoch_metric_variants,
    predict_cross_fitted_mean_templates,
    predict_cross_fitted_median_templates,
    predict_cross_fitted_ridge_templates,
    predict_cross_fitted_templates,
    prepare_beat_epochs,
)
from bcg_correction.bcg import (
    BcgCorrectionConfig,
    correct_bcg,
    rr_gap_spans,
)
from bcg_correction.bcg_config import DetectorConfig
from bcg_correction.cardiac import detect_r_peaks
from bcg_correction.figure_style import (
    ARTIFACT,
    COLLATERAL,
    CORRECTED,
    DASH,
    INK,
    MUTED,
    STYLE,
    UNCORRECTED,
    panel,
    save_figure,
)
from bcg_correction.metrics import (
    delay_estimation_eeg,
    estimate_ecg_to_bcg_delay,
    incremental_signal_transfer,
    is_posterior_eeg_channel,
    regress_out_reference,
    tone_transfer,
)

_EXPERIMENT_KEYS = frozenset(
    {
        "data_root",
        "output_root",
        "detector_config",
        "recording_glob",
        "subjects",
        "sampling_rate_hz",
        "correction_window_seconds",
        "cross_fit_fold_count",
        "ecg_window_seconds",
        "injection_amplitude_uv",
        "injection_event_count",
        "injection_event_width_seconds",
        "injection_tone_frequencies_hz",
        "neighbor_counts",
        "ridge_penalties",
        "morphology_component_counts",
        "robust_morphology_components",
        "morphology_samples",
        "aas_neighbor_count",
        "pca_obs_components",
        "null_surrogate_count",
        "random_seed",
    }
)
_DIAGNOSTIC_ONLY_METRICS = frozenset(
    {"specificity", "alpha_collateral_fraction"}
)


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Strict, reproducible settings for one multi-participant screen."""

    data_root: Path
    output_root: Path
    detector_config: Path
    recording_glob: str
    subjects: tuple[str, ...]
    sampling_rate_hz: float
    correction_window_seconds: tuple[float, float]
    cross_fit_fold_count: int
    ecg_window_seconds: tuple[float, float]
    injection_amplitude_uv: float
    injection_event_count: int
    injection_event_width_seconds: float
    injection_tone_frequencies_hz: tuple[float, ...]
    neighbor_counts: tuple[int, ...]
    ridge_penalties: tuple[float, ...]
    morphology_component_counts: tuple[int, ...]
    robust_morphology_components: int
    morphology_samples: int
    aas_neighbor_count: int
    pca_obs_components: int
    null_surrogate_count: int
    random_seed: int


@dataclass(frozen=True, slots=True)
class Candidate:
    """One feature representation and cross-fitted aggregator."""

    name: str
    family: str
    aggregator: str
    features: np.ndarray | None
    neighbor_count: int | None
    ridge_penalty: float | None


@dataclass(frozen=True, slots=True)
class Injection:
    """One known EEG signal added to a real recording."""

    name: str
    signal_uv: np.ndarray
    topography: np.ndarray
    temporal_uv: np.ndarray
    nominal_frequency_hz: float | None
    exact_frequency_hz: float | None


def _load_experiment_config(path: Path) -> ExperimentConfig:
    config_path = path.expanduser().resolve()
    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("experiment configuration must be a mapping")
    keys = frozenset(document)
    unknown = sorted(keys - _EXPERIMENT_KEYS)
    if unknown:
        raise ValueError(
            "experiment configuration has unknown keys: " + ", ".join(unknown)
        )
    missing = sorted(_EXPERIMENT_KEYS - keys)
    if missing:
        raise ValueError(
            "experiment configuration is missing keys: " + ", ".join(missing)
        )
    base = config_path.parent
    subjects = _subject_tuple(document["subjects"])
    neighbor_counts = _positive_integer_tuple(
        document["neighbor_counts"],
        name="neighbor_counts",
    )
    morphology_counts = _positive_integer_tuple(
        document["morphology_component_counts"],
        name="morphology_component_counts",
    )
    morphology_samples = _positive_integer(
        document["morphology_samples"],
        name="morphology_samples",
    )
    if morphology_counts[-1] > morphology_samples:
        raise ValueError(
            "morphology component counts cannot exceed morphology_samples"
        )
    robust_components = _positive_integer(
        document["robust_morphology_components"],
        name="robust_morphology_components",
    )
    if robust_components not in morphology_counts:
        raise ValueError(
            "robust_morphology_components must be in "
            "morphology_component_counts"
        )
    sampling_rate_hz = _positive_number(
        document["sampling_rate_hz"],
        name="sampling_rate_hz",
    )
    tone_frequencies = _positive_number_tuple(
        document["injection_tone_frequencies_hz"],
        name="injection_tone_frequencies_hz",
    )
    if tone_frequencies[-1] >= 0.5 * sampling_rate_hz:
        raise ValueError(
            "injection tone frequencies must stay below Nyquist"
        )
    return ExperimentConfig(
        data_root=_relative_path(document["data_root"], base, "data_root"),
        output_root=_relative_path(document["output_root"], base, "output_root"),
        detector_config=_relative_path(
            document["detector_config"],
            base,
            "detector_config",
        ),
        recording_glob=_recording_glob(document["recording_glob"]),
        subjects=subjects,
        sampling_rate_hz=sampling_rate_hz,
        correction_window_seconds=_interval(
            document["correction_window_seconds"],
            name="correction_window_seconds",
        ),
        cross_fit_fold_count=_positive_integer(
            document["cross_fit_fold_count"],
            name="cross_fit_fold_count",
        ),
        ecg_window_seconds=_interval(
            document["ecg_window_seconds"],
            name="ecg_window_seconds",
        ),
        injection_amplitude_uv=_positive_number(
            document["injection_amplitude_uv"],
            name="injection_amplitude_uv",
        ),
        injection_event_count=_positive_integer(
            document["injection_event_count"],
            name="injection_event_count",
        ),
        injection_event_width_seconds=_positive_number(
            document["injection_event_width_seconds"],
            name="injection_event_width_seconds",
        ),
        injection_tone_frequencies_hz=tone_frequencies,
        neighbor_counts=neighbor_counts,
        ridge_penalties=_positive_number_tuple(
            document["ridge_penalties"],
            name="ridge_penalties",
        ),
        morphology_component_counts=morphology_counts,
        robust_morphology_components=robust_components,
        morphology_samples=morphology_samples,
        aas_neighbor_count=_positive_integer(
            document["aas_neighbor_count"],
            name="aas_neighbor_count",
        ),
        pca_obs_components=_positive_integer(
            document["pca_obs_components"],
            name="pca_obs_components",
        ),
        null_surrogate_count=_positive_integer(
            document["null_surrogate_count"],
            name="null_surrogate_count",
        ),
        random_seed=_nonnegative_integer(
            document["random_seed"],
            name="random_seed",
        ),
    )


def _relative_path(value: object, base: Path, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty path string")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _recording_glob(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or Path(value).name != value
    ):
        raise ValueError("recording_glob must be one filename pattern")
    return value


def _subject_tuple(value: object) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or len(value) < 2
        or not all(isinstance(item, str) and item for item in value)
        or len(set(value)) != len(value)
    ):
        raise ValueError("subjects must contain at least two unique names")
    return tuple(value)


def _positive_integer_tuple(value: object, *, name: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a nonempty integer list")
    values = tuple(_positive_integer(item, name=name) for item in value)
    if any(left >= right for left, right in pairwise(values)):
        raise ValueError(f"{name} must be strictly increasing")
    return values


def _positive_number_tuple(value: object, *, name: str) -> tuple[float, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a nonempty number list")
    values = tuple(_positive_number(item, name=name) for item in value)
    if any(left >= right for left, right in pairwise(values)):
        raise ValueError(f"{name} must be strictly increasing")
    return values


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _nonnegative_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def _positive_number(value: object, *, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        or value <= 0.0
    ):
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _interval(value: object, *, name: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{name} must be a finite increasing pair")
    start, stop = value
    if (
        isinstance(start, bool)
        or isinstance(stop, bool)
        or not isinstance(start, Real)
        or not isinstance(stop, Real)
        or not math.isfinite(float(start))
        or not math.isfinite(float(stop))
        or start >= stop
    ):
        raise ValueError(f"{name} must be a finite increasing pair")
    return float(start), float(stop)


def _tone_injection(
    *,
    channel_count: int,
    sample_count: int,
    sampling_rate_hz: float,
    nominal_frequency_hz: float,
    amplitude_uv: float,
    generator: np.random.Generator,
) -> tuple[np.ndarray, float]:
    topography, temporal, exact_frequency = _tone_components(
        channel_count=channel_count,
        sample_count=sample_count,
        sampling_rate_hz=sampling_rate_hz,
        nominal_frequency_hz=nominal_frequency_hz,
        amplitude_uv=amplitude_uv,
        generator=generator,
    )
    return topography[:, np.newaxis] * temporal, exact_frequency


def _tone_components(
    *,
    channel_count: int,
    sample_count: int,
    sampling_rate_hz: float,
    nominal_frequency_hz: float,
    amplitude_uv: float,
    generator: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, float]:
    channels = _positive_integer(channel_count, name="channel_count")
    samples = _positive_integer(sample_count, name="sample_count")
    sampling_rate = _positive_number(
        sampling_rate_hz,
        name="sampling_rate_hz",
    )
    nominal_frequency = _positive_number(
        nominal_frequency_hz,
        name="nominal_frequency_hz",
    )
    amplitude = _positive_number(amplitude_uv, name="amplitude_uv")
    if nominal_frequency >= 0.5 * sampling_rate:
        raise ValueError("nominal_frequency_hz must stay below Nyquist")
    cycle_count = round(nominal_frequency * samples / sampling_rate)
    if cycle_count < 1:
        raise ValueError("recording is too short for the requested tone")
    exact_frequency = cycle_count * sampling_rate / samples
    topography = _random_topography(channels, generator)
    phase = generator.uniform(-np.pi, np.pi)
    angles = (
        2.0 * np.pi * cycle_count * np.arange(samples) / samples + phase
    )
    temporal = amplitude * np.sin(angles)
    return topography, temporal, exact_frequency


def _event_injection(
    *,
    channel_count: int,
    sample_count: int,
    sampling_rate_hz: float,
    event_count: int,
    width_seconds: float,
    amplitude_uv: float,
    generator: np.random.Generator,
) -> np.ndarray:
    topography, temporal = _event_components(
        channel_count=channel_count,
        sample_count=sample_count,
        sampling_rate_hz=sampling_rate_hz,
        event_count=event_count,
        width_seconds=width_seconds,
        amplitude_uv=amplitude_uv,
        generator=generator,
    )
    return topography[:, np.newaxis] * temporal


def _event_components(
    *,
    channel_count: int,
    sample_count: int,
    sampling_rate_hz: float,
    event_count: int,
    width_seconds: float,
    amplitude_uv: float,
    generator: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    channels = _positive_integer(channel_count, name="channel_count")
    samples = _positive_integer(sample_count, name="sample_count")
    sampling_rate = _positive_number(
        sampling_rate_hz,
        name="sampling_rate_hz",
    )
    events = _positive_integer(event_count, name="event_count")
    width = _positive_number(width_seconds, name="width_seconds")
    amplitude = _positive_number(amplitude_uv, name="amplitude_uv")
    half_width = math.ceil(4.0 * width * sampling_rate)
    if 2 * half_width + events > samples:
        raise ValueError("recording is too short for the requested events")

    temporal = np.zeros(samples, dtype=np.float64)
    edges = np.linspace(half_width, samples - half_width, events + 1)
    onsets = np.floor(
        generator.uniform(edges[:-1], edges[1:])
    ).astype(np.int64)
    offsets = np.arange(-half_width, half_width + 1)
    scaled_time = offsets / (width * sampling_rate)
    waveform = (1.0 - scaled_time**2) * np.exp(-0.5 * scaled_time**2)
    waveform /= np.max(np.abs(waveform))
    for onset in onsets:
        temporal[onset + offsets] += waveform
    topography = _random_topography(channels, generator)
    return topography, amplitude * temporal


def _random_topography(
    channel_count: int,
    generator: np.random.Generator,
) -> np.ndarray:
    if not isinstance(generator, np.random.Generator):
        raise TypeError("generator must be a NumPy Generator")
    topography = generator.normal(size=channel_count)
    return topography / np.sqrt(np.mean(topography**2))



def _participant_worker_count(
    *,
    participant_count: int,
    available_cpu_count: int,
    requested_count: int | None = None,
) -> int:
    for name, value in (
        ("participant_count", participant_count),
        ("available_cpu_count", available_cpu_count),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if requested_count is not None and (
        isinstance(requested_count, bool)
        or not isinstance(requested_count, int)
        or requested_count < 1
    ):
        raise ValueError("workers must be a positive integer")
    requested = available_cpu_count if requested_count is None else requested_count
    return min(participant_count, available_cpu_count, requested)


def _load_detector(path: Path) -> DetectorConfig:
    document = yaml.safe_load(path.expanduser().resolve().read_text())
    values = document["detector"]
    for key in ("preprocessing_band_hz", "template_window_seconds"):
        values[key] = tuple(values[key])
    return DetectorConfig(**values)


def _benchmark_participant(
    experiment: ExperimentConfig,
    subject: str,
    detector: DetectorConfig,
    *,
    seed: int,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, object],
]:
    vhdr = _recording_path(
        experiment.data_root,
        subject,
        experiment.recording_glob,
    )
    raw = mne.io.read_raw_brainvision(vhdr, preload=False, verbose="ERROR")
    try:
        posterior_names = tuple(
            name for name in raw.ch_names if is_posterior_eeg_channel(name)
        )
        if not posterior_names:
            raise RuntimeError(f"{subject} has no posterior EEG channels")
        if detector.ecg_channel not in raw.ch_names:
            raise RuntimeError(
                f"{subject} is missing ECG channel {detector.ecg_channel!r}"
            )
        selected_names = (*posterior_names, detector.ecg_channel)
        original = raw.get_data(picks=list(selected_names))
        original_sampling_rate = float(raw.info["sfreq"])
        original_sample_count = raw.n_times
    finally:
        raw.close()

    ecg = original[-1]
    detection = detect_r_peaks(
        ecg,
        original_sampling_rate,
        config=detector,
    )
    resampling_denominator = _integer_resampling_denominator(
        original_sampling_rate,
        experiment.sampling_rate_hz,
    )
    screened = resample_poly(original, 1, resampling_denominator, axis=1)
    screened_peaks = np.rint(
        detection.peak_samples / resampling_denominator
    ).astype(np.int64)
    if np.any(np.diff(screened_peaks) <= 0):
        raise RuntimeError(f"{subject} peaks collide after screening resampling")

    posterior = delay_estimation_eeg(
        screened,
        selected_names,
        ecg_channel_index=len(selected_names) - 1,
    )
    posterior -= np.median(posterior, axis=1, keepdims=True)
    delay_scan = estimate_ecg_to_bcg_delay(
        posterior,
        screened_peaks,
        sampling_rate_hz=experiment.sampling_rate_hz,
    )
    epoch_config = AdaptiveEpochConfig(
        correction_window_seconds=experiment.correction_window_seconds,
        ecg_window_seconds=experiment.ecg_window_seconds,
        ecg_to_bcg_delay_seconds=delay_scan.best_delay_seconds,
        morphology_components=max(experiment.morphology_component_counts),
        morphology_samples=experiment.morphology_samples,
    )
    prepared = prepare_beat_epochs(
        posterior,
        screened[-1],
        screened_peaks,
        experiment.sampling_rate_hz,
        config=epoch_config,
    )
    before_epochs = prepared.eeg_epochs * 1e6
    template_epochs = before_epochs - before_epochs.mean(axis=2, keepdims=True)
    training_mask = contiguous_cross_fit_training_mask(
        prepared.window_starts,
        epoch_samples=before_epochs.shape[2],
        fold_count=experiment.cross_fit_fold_count,
    )
    raw_locked, null_values = _heartbeat_locking_null(
        posterior * 1e6,
        prepared.window_starts,
        before_epochs.shape[2],
        surrogate_count=experiment.null_surrogate_count,
        seed=seed,
    )
    null_exceedance_probability = (
        1 + int(np.sum(null_values >= raw_locked))
    ) / (experiment.null_surrogate_count + 1)

    rows = []
    common = {
        "subject": subject,
        "input_vhdr": str(vhdr),
        "detector_status": detection.quality.status,
        "detected_beat_count": int(detection.peak_samples.size),
        "prepared_beat_count": int(prepared.peak_samples.size),
        "heart_rate_bpm": detection.quality.implied_rate_bpm,
        "rr_iqr_seconds": detection.quality.rr_iqr_seconds,
        "template_correlation_median": (
            detection.quality.template_correlation_median
        ),
        "delay_seconds": delay_scan.best_delay_seconds,
        "raw_locked_uv": raw_locked,
        "raw_null_median_uv": float(np.median(null_values)),
        "raw_null_exceedance_probability": null_exceedance_probability,
    }
    baseline_rows, bounded_base = _bounded_baseline_rows(
        posterior,
        screened[-1],
        metric_input_volts=screened,
        selected_names=(*posterior_names, "ECG"),
        peak_samples=screened_peaks,
        prepared_starts=prepared.window_starts,
        epoch_samples=before_epochs.shape[2],
        delay_seconds=delay_scan.best_delay_seconds,
        experiment=experiment,
        common=common,
    )
    rows.extend(baseline_rows)
    candidates = _candidates(prepared, experiment)
    for candidate in candidates:
        predicted = _predict_candidate(
            candidate,
            template_epochs,
            training_mask,
        )
        corrected_recording = apply_template_predictions_to_recording(
            posterior * 1e6,
            prepared.window_starts,
            predicted,
        )
        variants = _score_correction_variants(
            screened,
            corrected_recording,
            prepared.window_starts,
            epoch_samples=before_epochs.shape[2],
            sampling_rate_hz=experiment.sampling_rate_hz,
        )
        rows.append(
            common
            | _metric_variant_row(
                candidate.name,
                candidate.family,
                candidate.neighbor_count,
                candidate.aggregator,
                candidate.ridge_penalty,
                variants,
            )
        )
        del corrected_recording, predicted

    transfer_rows = _signal_transfer_rows(
        posterior,
        screened[-1],
        selected_names=(*posterior_names, "ECG"),
        peak_samples=screened_peaks,
        prepared_starts=prepared.window_starts,
        epoch_samples=before_epochs.shape[2],
        training_mask=training_mask,
        candidates=candidates,
        bounded_base=bounded_base,
        delay_seconds=delay_scan.best_delay_seconds,
        experiment=experiment,
        subject=subject,
        input_vhdr=vhdr,
        detector_status=detection.quality.status,
        seed=seed,
    )

    gap_samples = sum(
        stop - start
        for start, stop in rr_gap_spans(
            detection.peak_samples,
            original_sampling_rate,
            detector.maximum_rr_seconds,
        )
    )
    metadata = {
        "subject": subject,
        "input_vhdr": str(vhdr),
        "original_sampling_rate_hz": original_sampling_rate,
        "analysis_sampling_rate_hz": experiment.sampling_rate_hz,
        "original_sample_count": original_sample_count,
        "posterior_channels": list(posterior_names),
        "detector_status": detection.quality.status,
        "detector_quality": asdict(detection.quality),
        "detected_beat_count": int(detection.peak_samples.size),
        "prepared_beat_count": int(prepared.peak_samples.size),
        "minimum_cross_fit_training_beats": int(
            training_mask.sum(axis=1).min()
        ),
        "rr_gap_fraction": gap_samples / original_sample_count,
        "estimated_delay_seconds": delay_scan.best_delay_seconds,
        "raw_locked_uv": raw_locked,
        "raw_null_values_uv": null_values.tolist(),
    }
    del (
        original,
        screened,
        posterior,
        prepared,
        before_epochs,
        template_epochs,
        bounded_base,
    )
    gc.collect()
    return rows, transfer_rows, metadata


def _recording_path(
    data_root: Path,
    subject: str,
    recording_glob: str,
) -> Path:
    paths = tuple(
        path
        for path in (data_root / subject).glob(recording_glob)
        if not path.name.startswith("._")
    )
    if len(paths) != 1:
        raise RuntimeError(
            f"{subject} must have exactly one real {recording_glob!r} header; found "
            f"{len(paths)}"
        )
    return paths[0]


def _integer_resampling_denominator(
    original_sampling_rate: float,
    target_sampling_rate: float,
) -> int:
    denominator = round(original_sampling_rate / target_sampling_rate)
    if denominator < 1 or not np.isclose(
        original_sampling_rate / denominator,
        target_sampling_rate,
    ):
        raise RuntimeError(
            "screening requires an integer original-to-target sampling ratio"
        )
    return denominator


def _heartbeat_locking_null(
    eeg_uv: np.ndarray,
    window_starts: np.ndarray,
    epoch_samples: int,
    *,
    surrogate_count: int,
    seed: int,
) -> tuple[float, np.ndarray]:
    observed = _locked_amplitude(
        _extract_epochs(eeg_uv, window_starts, epoch_samples)
    )
    generator = np.random.default_rng(seed)
    null = np.empty(surrogate_count, dtype=np.float64)
    for index in range(surrogate_count):
        shift = int(generator.integers(1, eeg_uv.shape[1]))
        starts = (window_starts + shift) % eeg_uv.shape[1]
        complete = starts + epoch_samples <= eeg_uv.shape[1]
        if np.count_nonzero(complete) < 4:
            raise RuntimeError("a circular null retained fewer than four beats")
        null[index] = _locked_amplitude(
            _extract_epochs(eeg_uv, starts[complete], epoch_samples)
        )
    return observed, null


def _locked_amplitude(epochs: np.ndarray) -> float:
    template = epochs.mean(axis=0)
    return float(np.median(np.sqrt(np.mean(template**2, axis=1))))


def _extract_epochs(
    eeg: np.ndarray,
    starts: np.ndarray,
    epoch_samples: int,
) -> np.ndarray:
    indices = starts[:, np.newaxis] + np.arange(epoch_samples)
    return eeg[:, indices].transpose(1, 0, 2)


def _bounded_baseline_rows(
    posterior: np.ndarray,
    ecg: np.ndarray,
    *,
    metric_input_volts: np.ndarray,
    selected_names: tuple[str, ...],
    peak_samples: np.ndarray,
    prepared_starts: np.ndarray,
    epoch_samples: int,
    delay_seconds: float,
    experiment: ExperimentConfig,
    common: dict[str, object],
) -> tuple[list[dict[str, object]], dict[str, np.ndarray]]:
    rows = []
    corrected_by_method = {}
    settings = (
        (
            "aas",
            experiment.aas_neighbor_count,
            experiment.pca_obs_components,
        ),
        (
            "pca_obs",
            experiment.aas_neighbor_count,
            experiment.pca_obs_components,
        ),
    )
    for method, aas_neighbors, pca_components in settings:
        corrected_uv = _correct_bounded_posterior(
            posterior,
            ecg,
            selected_names=selected_names,
            peak_samples=peak_samples,
            delay_seconds=delay_seconds,
            experiment=experiment,
            method=method,
        )
        corrected_by_method[method] = corrected_uv
        variants = _score_correction_variants(
            metric_input_volts,
            corrected_uv,
            prepared_starts,
            epoch_samples=epoch_samples,
            sampling_rate_hz=experiment.sampling_rate_hz,
        )
        rows.append(
            common
            | _metric_variant_row(
                method,
                method,
                aas_neighbors if method == "aas" else pca_components,
                "target_fitted" if method == "aas" else "pca_obs",
                None,
                variants,
            )
        )
    return rows, corrected_by_method


def _score_correction_variants(
    selected_input_volts: np.ndarray,
    corrected_eeg_uv: np.ndarray,
    window_starts: np.ndarray,
    *,
    epoch_samples: int,
    sampling_rate_hz: float,
) -> ContinuousMetricVariants:
    selected = np.asarray(selected_input_volts, dtype=np.float64)
    corrected = np.asarray(corrected_eeg_uv, dtype=np.float64)
    if (
        selected.ndim != 2
        or selected.shape[0] != corrected.shape[0] + 1
        or selected.shape[1] != corrected.shape[1]
    ):
        raise ValueError(
            "selected_input_volts must contain corrected EEG channels then ECG"
        )
    return continuous_epoch_metric_variants(
        selected[:-1] * 1e6,
        corrected,
        selected[-1],
        window_starts,
        epoch_samples=epoch_samples,
        sampling_rate_hz=sampling_rate_hz,
    )


def _correct_bounded_posterior(
    posterior: np.ndarray,
    ecg: np.ndarray,
    *,
    selected_names: tuple[str, ...],
    peak_samples: np.ndarray,
    delay_seconds: float,
    experiment: ExperimentConfig,
    method: str,
) -> np.ndarray:
    data = np.vstack((posterior, ecg[np.newaxis, :]))
    correction = correct_bcg(
        data,
        peak_samples,
        experiment.sampling_rate_hz,
        channel_names=selected_names,
        eeg_picks=np.arange(posterior.shape[0], dtype=np.int64),
        ecg_channel_index=posterior.shape[0],
        config=BcgCorrectionConfig(
            method=method,
            window_seconds=experiment.correction_window_seconds,
            ecg_to_bcg_delay_seconds=delay_seconds,
            aas_neighbor_count=experiment.aas_neighbor_count,
            pca_obs_components=experiment.pca_obs_components,
        ),
    )
    return correction.data_volts[:-1] * 1e6


def _signal_transfer_rows(
    posterior: np.ndarray,
    ecg: np.ndarray,
    *,
    selected_names: tuple[str, ...],
    peak_samples: np.ndarray,
    prepared_starts: np.ndarray,
    epoch_samples: int,
    training_mask: np.ndarray,
    candidates: tuple[Candidate, ...],
    bounded_base: dict[str, np.ndarray],
    delay_seconds: float,
    experiment: ExperimentConfig,
    subject: str,
    input_vhdr: Path,
    detector_status: str,
    seed: int,
) -> list[dict[str, object]]:
    rows = []
    common = {
        "subject": subject,
        "input_vhdr": str(input_vhdr),
        "detector_status": detector_status,
        "injection_amplitude_uv": experiment.injection_amplitude_uv,
    }
    for injection in _injections(
        experiment,
        channel_count=posterior.shape[0],
        sample_count=posterior.shape[1],
        seed=seed,
    ):
        for method in ("aas", "pca_obs"):
            corrected_injected = _correct_bounded_posterior(
                posterior + injection.signal_uv * 1e-6,
                ecg,
                selected_names=selected_names,
                peak_samples=peak_samples,
                delay_seconds=delay_seconds,
                experiment=experiment,
                method=method,
            )
            increment = corrected_injected - bounded_base[method]
            rows.append(
                common
                | _transfer_metric_row(
                    method=method,
                    family=method,
                    neighbor_count=(
                        experiment.aas_neighbor_count
                        if method == "aas"
                        else experiment.pca_obs_components
                    ),
                    aggregator=(
                        "target_fitted" if method == "aas" else "pca_obs"
                    ),
                    ridge_penalty=None,
                    injection_name=injection.name,
                    injected_uv=injection.signal_uv,
                    corrected_increment_uv=increment,
                    reference_ecg=ecg,
                    sampling_rate_hz=experiment.sampling_rate_hz,
                    nominal_frequency_hz=injection.nominal_frequency_hz,
                    exact_frequency_hz=injection.exact_frequency_hz,
                )
            )
            del corrected_injected, increment

        for candidate in candidates:
            increment = _correct_candidate_injection(
                candidate,
                injection,
                prepared_starts=prepared_starts,
                epoch_samples=epoch_samples,
                training_mask=training_mask,
            )
            rows.append(
                common
                | _transfer_metric_row(
                    method=candidate.name,
                    family=candidate.family,
                    neighbor_count=candidate.neighbor_count,
                    aggregator=candidate.aggregator,
                    ridge_penalty=candidate.ridge_penalty,
                    injection_name=injection.name,
                    injected_uv=injection.signal_uv,
                    corrected_increment_uv=increment,
                    reference_ecg=ecg,
                    sampling_rate_hz=experiment.sampling_rate_hz,
                    nominal_frequency_hz=injection.nominal_frequency_hz,
                    exact_frequency_hz=injection.exact_frequency_hz,
                )
            )
            del increment
    return rows


def _correct_candidate_injection(
    candidate: Candidate,
    injection: Injection,
    *,
    prepared_starts: np.ndarray,
    epoch_samples: int,
    training_mask: np.ndarray,
) -> np.ndarray:
    temporal_recording = injection.temporal_uv[np.newaxis, :]
    signal_epochs = _extract_epochs(
        temporal_recording,
        prepared_starts,
        epoch_samples,
    )
    template_epochs = signal_epochs - signal_epochs.mean(
        axis=2,
        keepdims=True,
    )
    predicted = _predict_candidate(
        candidate,
        template_epochs,
        training_mask,
    )
    corrected_temporal = apply_template_predictions_to_recording(
        temporal_recording,
        prepared_starts,
        predicted,
    )[0]
    return injection.topography[:, np.newaxis] * corrected_temporal


def _injections(
    experiment: ExperimentConfig,
    *,
    channel_count: int,
    sample_count: int,
    seed: int,
):
    generator = np.random.default_rng(seed)
    for nominal_frequency in experiment.injection_tone_frequencies_hz:
        topography, temporal, exact_frequency = _tone_components(
            channel_count=channel_count,
            sample_count=sample_count,
            sampling_rate_hz=experiment.sampling_rate_hz,
            nominal_frequency_hz=nominal_frequency,
            amplitude_uv=experiment.injection_amplitude_uv,
            generator=generator,
        )
        signal = topography[:, np.newaxis] * temporal
        yield Injection(
            name=f"tone_{nominal_frequency:g}hz",
            signal_uv=signal,
            topography=topography,
            temporal_uv=temporal,
            nominal_frequency_hz=nominal_frequency,
            exact_frequency_hz=exact_frequency,
        )
    topography, temporal = _event_components(
        channel_count=channel_count,
        sample_count=sample_count,
        sampling_rate_hz=experiment.sampling_rate_hz,
        event_count=experiment.injection_event_count,
        width_seconds=experiment.injection_event_width_seconds,
        amplitude_uv=experiment.injection_amplitude_uv,
        generator=generator,
    )
    yield Injection(
        name="random_events",
        signal_uv=topography[:, np.newaxis] * temporal,
        topography=topography,
        temporal_uv=temporal,
        nominal_frequency_hz=None,
        exact_frequency_hz=None,
    )


def _candidates(
    prepared,
    experiment: ExperimentConfig,
) -> tuple[Candidate, ...]:
    groups = {
        "temporal": combine_feature_groups(prepared.temporal_features),
        "rhythm_time": combine_feature_groups(
            prepared.rhythm_features,
            prepared.temporal_features,
        ),
    }
    for component_count in experiment.morphology_component_counts:
        family = f"ecg{component_count:02d}_rhythm_time"
        groups[family] = combine_feature_groups(
            prepared.morphology_features[:, :component_count],
            prepared.rhythm_features,
            prepared.temporal_features,
        )
    candidates = []
    for family, features in groups.items():
        for neighbor_count in experiment.neighbor_counts:
            candidates.append(
                Candidate(
                    name=f"{family}_weighted_k{neighbor_count:02d}",
                    family=family,
                    neighbor_count=neighbor_count,
                    aggregator="weighted",
                    features=features,
                    ridge_penalty=None,
                )
            )
    robust_family = (
        f"ecg{experiment.robust_morphology_components:02d}_rhythm_time"
    )
    robust_features = groups[robust_family]
    for neighbor_count in experiment.neighbor_counts:
        candidates.append(
            Candidate(
                name=f"{robust_family}_median_k{neighbor_count:02d}",
                family=robust_family,
                neighbor_count=neighbor_count,
                aggregator="median",
                features=robust_features,
                ridge_penalty=None,
            )
        )
    for family, features in groups.items():
        for ridge_penalty in experiment.ridge_penalties:
            candidates.append(
                Candidate(
                    name=f"{family}_ridge_a{ridge_penalty:g}",
                    family=family,
                    aggregator="ridge",
                    features=features,
                    neighbor_count=None,
                    ridge_penalty=ridge_penalty,
                )
            )
    return tuple(candidates)


def _predict_candidate(
    candidate: Candidate,
    epochs: np.ndarray,
    training_mask: np.ndarray,
) -> np.ndarray:
    if candidate.aggregator == "mean":
        return predict_cross_fitted_mean_templates(epochs, training_mask)
    if candidate.aggregator == "weighted":
        return predict_cross_fitted_templates(
            epochs,
            candidate.features,
            training_mask,
            neighbor_count=candidate.neighbor_count,
        )
    if candidate.aggregator == "median":
        return predict_cross_fitted_median_templates(
            epochs,
            candidate.features,
            training_mask,
            neighbor_count=candidate.neighbor_count,
        )
    if candidate.aggregator == "ridge":
        return predict_cross_fitted_ridge_templates(
            epochs,
            candidate.features,
            training_mask,
            ridge_penalty=candidate.ridge_penalty,
        )
    raise ValueError(f"unknown candidate aggregator: {candidate.aggregator}")


def _metric_variant_row(
    method: str,
    family: str,
    neighbor_count: int | None,
    aggregator: str,
    ridge_penalty: float | None,
    variants: ContinuousMetricVariants,
) -> dict[str, object]:
    row: dict[str, object] = {
        "method": method,
        "family": family,
        "neighbor_count": neighbor_count,
        "aggregator": aggregator,
        "ridge_penalty": ridge_penalty,
    }
    for prefix, metrics in (
        ("direct", variants.direct),
        ("reference_orthogonalized", variants.reference_orthogonalized),
    ):
        for name, value in asdict(metrics).items():
            label = (
                f"diagnostic_{name}"
                if name in _DIAGNOSTIC_ONLY_METRICS
                else name
            )
            row[f"{prefix}_{label}"] = value
        if metrics.held_out_ratio <= 0.0:
            raise ValueError("held_out_ratio must be positive for log distortion")
        row[f"{prefix}_held_out_log_distortion"] = abs(
            math.log(metrics.held_out_ratio)
        )
    return row


def _transfer_metric_row(
    *,
    method: str,
    family: str,
    neighbor_count: int | None,
    aggregator: str,
    ridge_penalty: float | None,
    injection_name: str,
    injected_uv: np.ndarray,
    corrected_increment_uv: np.ndarray,
    reference_ecg: np.ndarray,
    sampling_rate_hz: float,
    nominal_frequency_hz: float | None,
    exact_frequency_hz: float | None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "method": method,
        "family": family,
        "neighbor_count": neighbor_count,
        "aggregator": aggregator,
        "ridge_penalty": ridge_penalty,
        "injection": injection_name,
        "nominal_frequency_hz": nominal_frequency_hz,
        "exact_frequency_hz": exact_frequency_hz,
    }
    for prefix, injected, corrected in (
        ("direct", injected_uv, corrected_increment_uv),
        (
            "reference_orthogonalized",
            regress_out_reference(injected_uv, reference_ecg),
            regress_out_reference(corrected_increment_uv, reference_ecg),
        ),
    ):
        transfer = incremental_signal_transfer(injected, corrected)
        if exact_frequency_hz is None:
            amplitude_ratio = None
            phase_error = None
        else:
            channel_transfers = tuple(
                tone_transfer(
                    injected_channel,
                    corrected_channel,
                    frequency=exact_frequency_hz,
                    sampling_rate=sampling_rate_hz,
                )
                for injected_channel, corrected_channel in zip(
                    injected,
                    corrected,
                    strict=True,
                )
            )
            amplitude_ratio = float(
                np.median(
                    [value.amplitude_ratio for value in channel_transfers]
                )
            )
            phase_error = float(
                np.median(
                    [
                        abs(value.phase_error_degrees)
                        for value in channel_transfers
                    ]
                )
            )
        row.update(
            {
                f"{prefix}_gain": transfer.gain,
                f"{prefix}_relative_error": transfer.relative_error,
                f"{prefix}_cosine_similarity": transfer.cosine_similarity,
                f"{prefix}_tone_amplitude_ratio_median": amplitude_ratio,
                f"{prefix}_tone_absolute_phase_error_degrees_median": phase_error,
            }
        )
    return row


def _summarize_cohort(
    participant_rows: list[dict[str, object]],
    *,
    seed: int,
) -> list[dict[str, object]]:
    methods = tuple(dict.fromkeys(str(row["method"]) for row in participant_rows))
    generator = np.random.default_rng(seed)
    summaries = []
    for method in methods:
        rows = [row for row in participant_rows if row["method"] == method]
        summary: dict[str, object] = {
            "method": method,
            "family": rows[0]["family"],
            "neighbor_count": rows[0]["neighbor_count"],
            "aggregator": rows[0]["aggregator"],
            "ridge_penalty": rows[0]["ridge_penalty"],
            "participant_count": len(rows),
        }
        metric_names = (
            *(
                f"diagnostic_{field.name}"
                if field.name in _DIAGNOSTIC_ONLY_METRICS
                else field.name
                for field in fields(EpochCorrectionMetrics)
            ),
            "held_out_log_distortion",
        )
        for prefix in ("direct", "reference_orthogonalized"):
            for metric_name in metric_names:
                column = f"{prefix}_{metric_name}"
                values = np.asarray([row[column] for row in rows], dtype=float)
                low, high = _bootstrap_median_interval(values, generator)
                summary[f"{column}_median"] = float(np.median(values))
                summary[f"{column}_q1"] = float(np.quantile(values, 0.25))
                summary[f"{column}_q3"] = float(np.quantile(values, 0.75))
                summary[f"{column}_bootstrap_low"] = low
                summary[f"{column}_bootstrap_high"] = high
        summaries.append(summary)
    return summaries


def _bootstrap_median_interval(
    values: np.ndarray,
    generator: np.random.Generator,
) -> tuple[float, float]:
    indices = generator.integers(0, values.size, size=(10_000, values.size))
    bootstrap = np.median(values[indices], axis=1)
    low, high = np.quantile(bootstrap, (0.025, 0.975))
    return float(low), float(high)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write an empty table: {path}")
    fieldnames = tuple(rows[0])
    if any(tuple(row) != fieldnames for row in rows):
        raise RuntimeError(f"rows are not rectangular: {path}")
    with path.open("x", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: object) -> None:
    with path.open("x", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2, default=_json_default)
        output_file.write("\n")


def _json_default(value: object) -> object:
    if isinstance(value, np.integer | np.floating):
        return value.item()
    raise TypeError(
        f"value is not JSON serializable: {type(value).__name__}"
    )


def _plot_metric_audit(rows: list[dict[str, object]], path: Path) -> None:
    """Show how the two symmetric ECG-handling variants change each metric."""
    family_colors = {
        "aas": ARTIFACT,
        "pca_obs": COLLATERAL,
        "temporal": MUTED,
        "rhythm_time": UNCORRECTED,
        "ecg04_rhythm_time": "#56B4E9",
        "ecg08_rhythm_time": CORRECTED,
        "ecg16_rhythm_time": "#E69F00",
    }
    with plt.rc_context(STYLE):
        figure, axes = plt.subplots(1, 2, figsize=(7.24, 3.55), layout="constrained")
        specifications = (
            (
                "locked_ratio",
                "heartbeat-locked residual ratio",
                "same metric; only ECG handling differs",
            ),
            (
                "held_out_log_distortion",
                "held-out distortion |log(ratio)|",
                "zero is preservation; attenuation is not rewarded",
            ),
        )
        for panel_index, (axis, (metric, title, subtitle)) in enumerate(
            zip(axes, specifications, strict=True)
        ):
            coordinates = []
            for row in rows:
                direct = float(row[f"direct_{metric}_median"])
                projected = float(
                    row[f"reference_orthogonalized_{metric}_median"]
                )
                coordinates.append((direct, projected))
                method = str(row["method"])
                axis.scatter(
                    direct,
                    projected,
                    color=family_colors[str(row["family"])],
                    marker={"mean": "^", "median": "D", "ridge": "s"}.get(
                        row["aggregator"],
                        "o",
                    ),
                    s=22,
                    alpha=0.9,
                    linewidths=0.35,
                    edgecolors="white",
                    zorder=3,
                )
                if method in {"aas", "pca_obs"}:
                    axis.annotate(
                        {"aas": "AAS", "pca_obs": "PCA-OBS"}[method],
                        (direct, projected),
                        xytext=(4, 4),
                        textcoords="offset points",
                        fontsize=6,
                        color=INK,
                    )
            bounds = np.asarray(coordinates)
            lower = float(np.min(bounds))
            upper = float(np.max(bounds))
            margin = max(0.02 * (upper - lower), 1e-6)
            identity = (lower - margin, upper + margin)
            axis.plot(identity, identity, color=MUTED, lw=0.7, ls=DASH, zorder=1)
            axis.set_xlabel(f"direct {title}")
            axis.set_ylabel(f"ECG-orthogonalized {title}")
            panel(axis, chr(65 + panel_index), title, subtitle)
        figure.suptitle(
            "Metric audit — no method-selection flag is computed",
            fontsize=9.0,
            fontweight="bold",
            x=0.0,
            ha="left",
            color=INK,
        )
        save_figure(figure, path)
        plt.close(figure)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workers", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    experiment = _load_experiment_config(arguments.config)
    if not experiment.data_root.is_dir():
        raise NotADirectoryError(
            f"data root does not exist: {experiment.data_root}"
        )
    experiment.output_root.mkdir(parents=True, exist_ok=False)

    detector = _load_detector(experiment.detector_config)
    available_cpu_count = os.cpu_count()
    if available_cpu_count is None:
        raise RuntimeError("operating system did not report an available CPU count")
    worker_count = _participant_worker_count(
        participant_count=len(experiment.subjects),
        available_cpu_count=available_cpu_count,
        requested_count=arguments.workers,
    )
    participant_rows: list[dict[str, object]] = []
    transfer_rows: list[dict[str, object]] = []
    participant_metadata: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = tuple(
            executor.submit(
                _benchmark_participant,
                experiment,
                subject,
                detector,
                seed=experiment.random_seed + participant_index,
            )
            for participant_index, subject in enumerate(experiment.subjects)
        )
        for subject, future in zip(experiment.subjects, futures, strict=True):
            rows, participant_transfers, metadata = future.result()
            participant_rows.extend(rows)
            transfer_rows.extend(participant_transfers)
            participant_metadata.append(metadata)
            print(
                f"{subject}: {metadata['prepared_beat_count']} beats, "
                f"{len(rows)} methods, detector={metadata['detector_status']}",
                flush=True,
            )

    summary_rows = _summarize_cohort(
        participant_rows,
        seed=experiment.random_seed,
    )
    _write_csv(
        experiment.output_root / "participant_metrics.csv",
        participant_rows,
    )
    _write_csv(experiment.output_root / "signal_transfer.csv", transfer_rows)
    _write_csv(experiment.output_root / "cohort_summary.csv", summary_rows)
    _write_json(
        experiment.output_root / "run.json",
        {
            "config_path": str(arguments.config.expanduser().resolve()),
            "data_root": str(experiment.data_root),
            "recording_glob": experiment.recording_glob,
            "subjects": list(experiment.subjects),
            "sampling_rate_hz": experiment.sampling_rate_hz,
            "correction_window_seconds": (
                experiment.correction_window_seconds
            ),
            "cross_fit_fold_count": experiment.cross_fit_fold_count,
            "ecg_window_seconds": experiment.ecg_window_seconds,
            "injection_amplitude_uv": experiment.injection_amplitude_uv,
            "injection_event_count": experiment.injection_event_count,
            "injection_event_width_seconds": (
                experiment.injection_event_width_seconds
            ),
            "injection_tone_frequencies_hz": (
                experiment.injection_tone_frequencies_hz
            ),
            "neighbor_counts": experiment.neighbor_counts,
            "ridge_penalties": experiment.ridge_penalties,
            "morphology_component_counts": (
                experiment.morphology_component_counts
            ),
            "robust_morphology_components": (
                experiment.robust_morphology_components
            ),
            "morphology_samples": experiment.morphology_samples,
            "aas_neighbor_count": experiment.aas_neighbor_count,
            "pca_obs_components": experiment.pca_obs_components,
            "null_surrogate_count": experiment.null_surrogate_count,
            "random_seed": experiment.random_seed,
            "participant_worker_count": worker_count,
            "available_cpu_count": available_cpu_count,
            "participants": participant_metadata,
        },
    )
    _plot_metric_audit(
        summary_rows,
        experiment.output_root / "cohort_metric_audit.png",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
