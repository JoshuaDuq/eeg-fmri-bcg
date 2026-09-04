"""Cross-fitted heartbeat templates predicted without target-beat EEG."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
import numpy.typing as npt
from scipy.linalg.blas import dgemm, dgemv
from scipy.signal import butter, sosfiltfilt, welch

from bcg_correction.metrics import regress_out_reference


class AdaptiveInputError(ValueError):
    """Raised when adaptive template inputs cannot form valid predictions."""


def _validate_finite_number(value: object, *, name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
    ):
        raise AdaptiveInputError(f"{name} must contain finite numbers")


def _validate_positive_integer(value: object, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise AdaptiveInputError(f"{name} must be a positive integer")


def _validate_interval(values: object, *, name: str) -> None:
    if not isinstance(values, tuple) or len(values) != 2:
        raise AdaptiveInputError(f"{name} must be a finite increasing pair")
    for value in values:
        _validate_finite_number(value, name=name)
    if values[0] >= values[1]:
        raise AdaptiveInputError(f"{name} must be a finite increasing pair")


@dataclass(frozen=True, slots=True)
class AdaptiveEpochConfig:
    """Windows and ECG representation used by an adaptive experiment."""

    correction_window_seconds: tuple[float, float]
    ecg_window_seconds: tuple[float, float]
    ecg_to_bcg_delay_seconds: float
    morphology_components: int
    morphology_samples: int
    morphology_band_hz: tuple[float, float] = (0.5, 20.0)

    def __post_init__(self) -> None:
        _validate_interval(
            self.correction_window_seconds,
            name="correction_window_seconds",
        )
        _validate_interval(
            self.ecg_window_seconds,
            name="ecg_window_seconds",
        )
        _validate_interval(self.morphology_band_hz, name="morphology_band_hz")
        _validate_finite_number(
            self.ecg_to_bcg_delay_seconds,
            name="ecg_to_bcg_delay_seconds",
        )
        _validate_positive_integer(
            self.morphology_components,
            name="morphology_components",
        )
        _validate_positive_integer(
            self.morphology_samples,
            name="morphology_samples",
        )


@dataclass(frozen=True, slots=True)
class PreparedBeatEpochs:
    """Common beat windows and independent conditioning feature groups."""

    eeg_epochs: npt.NDArray[np.float64]
    peak_samples: npt.NDArray[np.int64]
    window_starts: npt.NDArray[np.int64]
    temporal_features: npt.NDArray[np.float64]
    rhythm_features: npt.NDArray[np.float64]
    morphology_features: npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class EpochCorrectionMetrics:
    """Suppression and collateral measurements for common beat epochs."""

    locked_ratio: float
    held_out_ratio: float
    locked_before: float
    locked_after: float
    specificity: float
    alpha_collateral_fraction: float
    locked_removed: float
    collateral: float


@dataclass(frozen=True, slots=True)
class ContinuousMetricVariants:
    """Direct and reference-orthogonalized views of one correction."""

    direct: EpochCorrectionMetrics
    reference_orthogonalized: EpochCorrectionMetrics


def prepare_beat_epochs(
    eeg: npt.ArrayLike,
    ecg: npt.ArrayLike,
    peak_samples: npt.ArrayLike,
    sampling_rate_hz: float,
    *,
    config: AdaptiveEpochConfig,
) -> PreparedBeatEpochs:
    """Extract one common beat table for every experimental predictor."""
    if not isinstance(config, AdaptiveEpochConfig):
        raise AdaptiveInputError("config must be an AdaptiveEpochConfig")
    recording = _validate_recording(eeg)
    reference = _validate_ecg(ecg, recording.shape[1])
    peaks = _validate_peaks(peak_samples, recording.shape[1])
    sampling_rate = _validate_sampling_rate(sampling_rate_hz)
    _, high_hz = config.morphology_band_hz
    if high_hz >= 0.5 * sampling_rate:
        raise AdaptiveInputError(
            "morphology_band_hz must stay below the Nyquist frequency"
        )

    correction_offsets = _window_offsets(
        config.correction_window_seconds,
        sampling_rate,
        delay_seconds=config.ecg_to_bcg_delay_seconds,
    )
    ecg_offsets = _window_offsets(
        config.ecg_window_seconds,
        sampling_rate,
        delay_seconds=0.0,
    )
    complete = _complete_indices(
        peaks,
        sample_count=recording.shape[1],
        windows=(correction_offsets, ecg_offsets),
    )
    if complete.size <= config.morphology_components:
        raise AdaptiveInputError(
            "morphology_components must be smaller than the complete beat count"
        )
    correction_start, correction_stop = correction_offsets
    window_starts = peaks[complete] + correction_start
    sample_offsets = np.arange(correction_stop - correction_start)
    eeg_indices = window_starts[:, np.newaxis] + sample_offsets
    eeg_epochs = recording[:, eeg_indices].transpose(1, 0, 2)

    filtered_ecg = _bandpass(
        reference,
        sampling_rate,
        config.morphology_band_hz,
    )
    ecg_start, ecg_stop = ecg_offsets
    morphology_starts = peaks[complete] + ecg_start
    morphology_indices = morphology_starts[:, np.newaxis] + np.arange(
        ecg_stop - ecg_start
    )
    morphology_epochs = filtered_ecg[morphology_indices]
    morphology_features = _morphology_scores(
        morphology_epochs,
        component_count=config.morphology_components,
        output_samples=config.morphology_samples,
    )
    kept_peaks = peaks[complete]
    return PreparedBeatEpochs(
        eeg_epochs=np.asarray(eeg_epochs, dtype=np.float64),
        peak_samples=kept_peaks,
        window_starts=window_starts,
        temporal_features=(kept_peaks / sampling_rate)[:, np.newaxis],
        rhythm_features=_rhythm_features(peaks, complete, sampling_rate),
        morphology_features=morphology_features,
    )


def combine_feature_groups(*feature_groups: npt.ArrayLike) -> np.ndarray:
    """Robustly standardize groups so dimensionality cannot dominate distance."""
    if not feature_groups:
        raise AdaptiveInputError("at least one feature group is required")
    groups = [np.asarray(group) for group in feature_groups]
    row_count = groups[0].shape[0] if groups[0].ndim == 2 else -1
    standardized = []
    for group in groups:
        values = _validate_features(group, row_count)
        centred = values - np.median(values, axis=0, keepdims=True)
        scale = 1.4826 * np.median(np.abs(centred), axis=0, keepdims=True)
        nonconstant = scale[0] > 0.0
        transformed = np.zeros_like(centred)
        transformed[:, nonconstant] = (
            centred[:, nonconstant] / scale[:, nonconstant]
        )
        transformed /= math.sqrt(values.shape[1])
        standardized.append(transformed)
    return np.concatenate(standardized, axis=1)


def apply_template_predictions_to_recording(
    eeg: npt.ArrayLike,
    window_starts: npt.ArrayLike,
    predicted_templates: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Apply cross-fitted templates on one physically consistent timeline."""
    recording = _validate_recording(eeg)
    predictions = _validate_epochs(predicted_templates)
    starts = np.asarray(window_starts)
    if (
        starts.ndim != 1
        or starts.size != predictions.shape[0]
        or np.issubdtype(starts.dtype, np.bool_)
        or not np.issubdtype(starts.dtype, np.integer)
    ):
        raise AdaptiveInputError(
            "window_starts must contain one integer per predicted template"
        )
    starts = starts.astype(np.int64, copy=False)
    if predictions.shape[1] != recording.shape[0]:
        raise AdaptiveInputError(
            "predicted_templates must contain one row per EEG channel"
        )
    stops = starts + predictions.shape[2]
    if (
        np.any(starts < 0)
        or np.any(stops > recording.shape[1])
        or np.any(np.diff(starts) <= 0)
    ):
        raise AdaptiveInputError(
            "window_starts must be strictly increasing complete windows"
        )

    tapered = predictions * _cosine_taper(predictions.shape[2])
    correction_sum = np.zeros_like(recording)
    correction_count = np.zeros(recording.shape[1], dtype=np.int64)
    for start, stop, template in zip(
        starts,
        stops,
        tapered,
        strict=True,
    ):
        correction_sum[:, start:stop] += template
        correction_count[start:stop] += 1
    corrected = recording.copy()
    covered = correction_count > 0
    corrected[:, covered] -= (
        correction_sum[:, covered] / correction_count[covered]
    )
    return corrected


def epoch_correction_metrics(
    before_epochs: npt.ArrayLike,
    after_epochs: npt.ArrayLike,
    *,
    sampling_rate_hz: float,
    alpha_band_hz: tuple[float, float] = (8.0, 13.0),
) -> EpochCorrectionMetrics:
    """Measure residual locking and non-locked removal on shared beat windows.

    Both epoch stacks must already carry the same per-channel DC. A constant
    offset is perfectly heartbeat-locked -- it survives beat averaging intact --
    so a corrector that shifts a channel's baseline has that offset counted as
    residual artifact and can score close to 1.0 while having removed the
    artifact well. ``continuous_epoch_metric_variants`` centres both signals
    before epoching for exactly this reason, and ``cardiac_locked_rms`` strips
    the recording-level median for the same one; prefer either to calling this
    with epochs cut straight out of a corrected recording.
    """
    before = _validate_epochs(before_epochs)
    after = _validate_epochs(after_epochs)
    if after.shape != before.shape:
        raise AdaptiveInputError(
            "after_epochs must have the same shape as before_epochs"
        )
    sampling_rate = _validate_sampling_rate(sampling_rate_hz)
    _validate_interval(alpha_band_hz, name="alpha_band_hz")
    if alpha_band_hz[1] >= 0.5 * sampling_rate:
        raise AdaptiveInputError(
            "alpha_band_hz must stay below the Nyquist frequency"
        )

    locked_before_by_channel = _root_mean_square(before.mean(axis=0))
    locked_after_by_channel = _root_mean_square(after.mean(axis=0))
    held_out_before = _held_out_rms(before)
    held_out_after = _held_out_rms(after)
    if np.any(held_out_before == 0.0):
        raise AdaptiveInputError(
            "before_epochs contain a channel with no held-out variation"
        )
    locked_before = float(np.median(locked_before_by_channel))
    locked_after = float(np.median(locked_after_by_channel))
    if locked_before == 0.0:
        raise AdaptiveInputError("before_epochs have no heartbeat-locked energy")

    removed = before - after
    removed_locked = removed.mean(axis=0)
    removed_nonlocked = removed - removed_locked[np.newaxis, :, :]
    total_removed = np.sqrt(np.mean(removed**2, axis=(0, 2)))
    locked_removed_by_channel = _root_mean_square(removed_locked)
    with np.errstate(divide="ignore", invalid="ignore"):
        channel_specificity = np.divide(
            locked_removed_by_channel,
            total_removed,
            out=np.full_like(total_removed, np.nan),
            where=total_removed > 0.0,
        )
    specificity = float(np.nanmedian(channel_specificity))
    collateral_by_channel = np.sqrt(
        np.mean(removed_nonlocked**2, axis=(0, 2))
    )
    raw_alpha = _band_power(before, sampling_rate, alpha_band_hz)
    collateral_alpha = _band_power(
        removed_nonlocked,
        sampling_rate,
        alpha_band_hz,
    )
    return EpochCorrectionMetrics(
        locked_ratio=locked_after / locked_before,
        held_out_ratio=float(np.median(held_out_after / held_out_before)),
        locked_before=locked_before,
        locked_after=locked_after,
        specificity=specificity,
        alpha_collateral_fraction=(
            collateral_alpha / raw_alpha if raw_alpha > 0.0 else float("nan")
        ),
        locked_removed=float(np.median(locked_removed_by_channel)),
        collateral=float(np.median(collateral_by_channel)),
    )


def continuous_epoch_metric_variants(
    before_eeg: npt.ArrayLike,
    after_eeg: npt.ArrayLike,
    reference: npt.ArrayLike,
    window_starts: npt.ArrayLike,
    *,
    epoch_samples: int,
    sampling_rate_hz: float,
) -> ContinuousMetricVariants:
    """Score one correction with neither and with both sides projected."""
    before = _validate_recording(before_eeg)
    after = _validate_recording(after_eeg)
    if after.shape != before.shape:
        raise AdaptiveInputError(
            "after_eeg must have the same shape as before_eeg"
        )
    cardiac_reference = _validate_ecg(reference, before.shape[1])
    starts = np.asarray(window_starts)
    if (
        starts.ndim != 1
        or starts.size < 4
        or np.issubdtype(starts.dtype, np.bool_)
        or not np.issubdtype(starts.dtype, np.integer)
    ):
        raise AdaptiveInputError(
            "window_starts must contain at least four integer samples"
        )
    starts = starts.astype(np.int64, copy=False)
    _validate_positive_integer(epoch_samples, name="epoch_samples")
    if (
        np.any(starts < 0)
        or np.any(np.diff(starts) <= 0)
        or np.any(starts + epoch_samples > before.shape[1])
    ):
        raise AdaptiveInputError(
            "window_starts must be increasing complete windows"
        )
    sampling_rate = _validate_sampling_rate(sampling_rate_hz)
    indices = starts[:, np.newaxis] + np.arange(epoch_samples)

    def score(left: np.ndarray, right: np.ndarray) -> EpochCorrectionMetrics:
        centred_left = left - np.median(left, axis=1, keepdims=True)
        centred_right = right - np.median(right, axis=1, keepdims=True)
        before_epochs = centred_left[:, indices].transpose(1, 0, 2)
        after_epochs = centred_right[:, indices].transpose(1, 0, 2)
        return epoch_correction_metrics(
            before_epochs,
            after_epochs,
            sampling_rate_hz=sampling_rate,
        )

    return ContinuousMetricVariants(
        direct=score(before, after),
        reference_orthogonalized=score(
            regress_out_reference(before, cardiac_reference),
            regress_out_reference(after, cardiac_reference),
        ),
    )


def contiguous_cross_fit_training_mask(
    window_starts: npt.ArrayLike,
    *,
    epoch_samples: int,
    fold_count: int,
) -> npt.NDArray[np.bool_]:
    """Return eligible training beats for contiguous held-out blocks.

    Beats in the target's contiguous fold are excluded. Windows in other folds
    are also excluded when they physically overlap the target window, so the
    predictor cannot see any EEG sample that it will later correct.
    """
    starts = np.asarray(window_starts)
    if (
        starts.ndim != 1
        or starts.size < 2
        or np.issubdtype(starts.dtype, np.bool_)
        or not np.issubdtype(starts.dtype, np.integer)
    ):
        raise AdaptiveInputError(
            "window_starts must contain at least two integer samples"
        )
    starts = starts.astype(np.int64, copy=False)
    if np.any(starts < 0) or np.any(np.diff(starts) <= 0):
        raise AdaptiveInputError(
            "window_starts must be finite, nonnegative, and strictly increasing"
        )
    _validate_positive_integer(epoch_samples, name="epoch_samples")
    _validate_positive_integer(fold_count, name="fold_count")
    if fold_count < 2 or fold_count > starts.size:
        raise AdaptiveInputError(
            "fold_count must be between two and the beat count"
        )

    folds = np.empty(starts.size, dtype=np.int64)
    for fold, indices in enumerate(np.array_split(np.arange(starts.size), fold_count)):
        folds[indices] = fold
    same_fold = folds[:, np.newaxis] == folds[np.newaxis, :]
    stops = starts + int(epoch_samples)
    overlaps = (starts[:, np.newaxis] < stops[np.newaxis, :]) & (
        starts[np.newaxis, :] < stops[:, np.newaxis]
    )
    return ~(same_fold | overlaps)


def predict_cross_fitted_templates(
    eeg_epochs: npt.ArrayLike,
    conditioning_features: npt.ArrayLike,
    training_mask: npt.ArrayLike,
    *,
    neighbor_count: int,
) -> npt.NDArray[np.float64]:
    """Predict every EEG epoch from feature-nearest *other* epochs.

    The features may describe ECG morphology, rhythm, or acquisition time, but
    must not contain EEG from the target epoch. Exact nearest neighbours are
    used without a similarity cutoff; this keeps weak-but-potentially-useful
    configurations measurable instead of discarding them by threshold.
    """
    epochs = _validate_epochs(eeg_epochs)
    features = _validate_features(conditioning_features, epochs.shape[0])
    eligible = _validate_training_mask(training_mask, epochs.shape[0])
    neighbors = _validate_neighbor_count(neighbor_count, epochs.shape[0])
    if np.any(np.sum(eligible, axis=1) < neighbors):
        raise AdaptiveInputError(
            "training_mask leaves fewer eligible beats than neighbor_count"
        )

    predictions = np.empty_like(epochs)
    for target_index, target_features in enumerate(features):
        distances, indices = _nearest_neighbors(
            features,
            target_features,
            eligible[target_index],
            neighbors,
        )
        weights = _adaptive_weights(distances[indices])
        predictions[target_index] = np.average(
            epochs[indices],
            axis=0,
            weights=weights,
        )
    return predictions


def predict_cross_fitted_mean_templates(
    eeg_epochs: npt.ArrayLike,
    training_mask: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Predict epochs by the mean of all eligible training beats."""
    epochs = _validate_epochs(eeg_epochs)
    eligible = _validate_training_mask(training_mask, epochs.shape[0])
    predictions = np.empty_like(epochs)
    template_cache: dict[bytes, np.ndarray] = {}
    for target_index, target_eligibility in enumerate(eligible):
        training_indices = np.flatnonzero(target_eligibility)
        if training_indices.size == 0:
            raise AdaptiveInputError(
                "training_mask leaves no eligible mean-template beats"
            )
        cache_key = training_indices.tobytes()
        template = template_cache.get(cache_key)
        if template is None:
            template = epochs[training_indices].mean(axis=0)
            template_cache[cache_key] = template
        predictions[target_index] = template
    return predictions


def predict_cross_fitted_reference_residual_mean_templates(
    eeg_epochs: npt.ArrayLike,
    reference_epochs: npt.ArrayLike,
    training_mask: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Predict mean templates after training-only reference regression.

    The regression and template are both estimated independently for each
    eligible training set. EEG from a held-out block therefore cannot affect
    either stage of its own prediction.

    Experiment variant. The template is ECG-free, so subtracting it leaves the
    target beat's own volume-conducted ECG in the output.
    """
    epochs = _validate_epochs(eeg_epochs)
    reference = np.asarray(reference_epochs)
    if (
        reference.ndim != 2
        or reference.shape != (epochs.shape[0], epochs.shape[2])
        or np.issubdtype(reference.dtype, np.bool_)
        or not np.issubdtype(reference.dtype, np.number)
    ):
        raise AdaptiveInputError(
            "reference_epochs must contain one numeric trace per EEG epoch"
        )
    reference = reference.astype(np.float64, copy=False)
    if not np.all(np.isfinite(reference)):
        raise AdaptiveInputError("reference_epochs must be finite")
    eligible = _validate_training_mask(training_mask, epochs.shape[0])

    predictions = np.empty_like(epochs)
    template_cache: dict[bytes, np.ndarray] = {}
    for target_index, target_eligibility in enumerate(eligible):
        training_indices = np.flatnonzero(target_eligibility)
        if training_indices.size == 0:
            raise AdaptiveInputError(
                "training_mask leaves no eligible mean-template beats"
            )
        cache_key = training_indices.tobytes()
        template = template_cache.get(cache_key)
        if template is None:
            training = epochs[training_indices]
            channel_first = training.transpose(1, 0, 2)
            flattened = channel_first.reshape(epochs.shape[1], -1)
            residual = regress_out_reference(
                flattened,
                reference[training_indices].reshape(-1),
            )
            residual_epochs = residual.reshape(channel_first.shape).transpose(
                1,
                0,
                2,
            )
            residual_epochs -= residual_epochs.mean(axis=2, keepdims=True)
            template = residual_epochs.mean(axis=0)
            template_cache[cache_key] = template
        predictions[target_index] = template
    return predictions


def predict_cross_fitted_median_templates(
    eeg_epochs: npt.ArrayLike,
    conditioning_features: npt.ArrayLike,
    training_mask: npt.ArrayLike,
    *,
    neighbor_count: int,
) -> npt.NDArray[np.float64]:
    """Predict each epoch by the median of its feature-nearest other epochs."""
    epochs = _validate_epochs(eeg_epochs)
    features = _validate_features(conditioning_features, epochs.shape[0])
    eligible = _validate_training_mask(training_mask, epochs.shape[0])
    neighbors = _validate_neighbor_count(neighbor_count, epochs.shape[0])
    if np.any(np.sum(eligible, axis=1) < neighbors):
        raise AdaptiveInputError(
            "training_mask leaves fewer eligible beats than neighbor_count"
        )

    predictions = np.empty_like(epochs)
    for target_index, target_features in enumerate(features):
        _, indices = _nearest_neighbors(
            features,
            target_features,
            eligible[target_index],
            neighbors,
        )
        predictions[target_index] = np.median(epochs[indices], axis=0)
    return predictions


def predict_cross_fitted_ridge_templates(
    eeg_epochs: npt.ArrayLike,
    conditioning_features: npt.ArrayLike,
    training_mask: npt.ArrayLike,
    *,
    ridge_penalty: float,
) -> npt.NDArray[np.float64]:
    """Predict epochs by ridge models fitted only to eligible beats.

    Coefficients are reused for targets with identical training masks. The
    intercept is unpenalized; conditioning columns should be standardized.
    """
    epochs = _validate_epochs(eeg_epochs)
    features = _validate_features(conditioning_features, epochs.shape[0])
    eligible = _validate_training_mask(training_mask, epochs.shape[0])
    if (
        isinstance(ridge_penalty, bool)
        or not isinstance(ridge_penalty, Real)
        or not math.isfinite(float(ridge_penalty))
        or ridge_penalty <= 0.0
    ):
        raise AdaptiveInputError("ridge_penalty must be finite and positive")

    design = np.column_stack((np.ones(features.shape[0]), features))
    targets = epochs.reshape(epochs.shape[0], -1)
    penalty = np.eye(design.shape[1]) * float(ridge_penalty)
    penalty[0, 0] = 0.0
    predictions = np.empty_like(targets)
    coefficient_cache: dict[bytes, np.ndarray] = {}
    for target_index, target_eligibility in enumerate(eligible):
        training_indices = np.flatnonzero(target_eligibility)
        if training_indices.size < 2:
            raise AdaptiveInputError(
                "training_mask leaves fewer than two ridge training beats"
            )
        cache_key = training_indices.tobytes()
        coefficients = coefficient_cache.get(cache_key)
        if coefficients is None:
            training_design = design[training_indices]
            training_targets = targets[training_indices]
            gram = dgemm(
                1.0,
                training_design,
                training_design,
                trans_a=True,
            ) + penalty
            right_hand_side = dgemm(
                1.0,
                training_design,
                training_targets,
                trans_a=True,
            )
            coefficients = np.linalg.solve(
                gram,
                right_hand_side,
            )
            coefficient_cache[cache_key] = coefficients
        predictions[target_index] = dgemv(
            1.0,
            coefficients,
            design[target_index],
            trans=1,
        )
    return predictions.reshape(epochs.shape)


def _nearest_neighbors(
    features: np.ndarray,
    target_features: np.ndarray,
    eligibility: np.ndarray,
    neighbor_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    distances = np.linalg.norm(features - target_features, axis=1)
    distances[~eligibility] = np.inf
    indices = np.argsort(distances, kind="stable")[:neighbor_count]
    return distances, indices


def _validate_epochs(eeg_epochs: npt.ArrayLike) -> np.ndarray:
    epochs = np.asarray(eeg_epochs)
    if epochs.ndim != 3 or min(epochs.shape) < 1:
        raise AdaptiveInputError(
            "eeg_epochs must have shape (beats, channels, samples)"
        )
    if np.issubdtype(epochs.dtype, np.bool_) or not np.issubdtype(
        epochs.dtype,
        np.number,
    ):
        raise AdaptiveInputError("eeg_epochs must contain finite numbers")
    epochs = epochs.astype(np.float64, copy=False)
    if not np.all(np.isfinite(epochs)):
        raise AdaptiveInputError("eeg_epochs must contain finite numbers")
    return epochs


def _validate_features(
    conditioning_features: npt.ArrayLike,
    beat_count: int,
) -> np.ndarray:
    features = np.asarray(conditioning_features)
    if features.ndim != 2 or features.shape[0] != beat_count:
        raise AdaptiveInputError(
            "conditioning_features must have one row per EEG epoch"
        )
    if features.shape[1] < 1:
        raise AdaptiveInputError(
            "conditioning_features must contain at least one feature"
        )
    if np.issubdtype(features.dtype, np.bool_) or not np.issubdtype(
        features.dtype,
        np.number,
    ):
        raise AdaptiveInputError(
            "conditioning_features must contain finite numbers"
        )
    features = features.astype(np.float64, copy=False)
    if not np.all(np.isfinite(features)):
        raise AdaptiveInputError(
            "conditioning_features must contain finite numbers"
        )
    return features


def _validate_training_mask(
    training_mask: npt.ArrayLike,
    beat_count: int,
) -> np.ndarray:
    mask = np.asarray(training_mask)
    if mask.shape != (beat_count, beat_count) or not np.issubdtype(
        mask.dtype,
        np.bool_,
    ):
        raise AdaptiveInputError(
            "training_mask must be a boolean beat-by-beat matrix"
        )
    if np.any(np.diag(mask)):
        raise AdaptiveInputError("training_mask cannot include a target beat")
    return mask.astype(bool, copy=False)


def _validate_neighbor_count(neighbor_count: int, beat_count: int) -> int:
    if (
        isinstance(neighbor_count, bool)
        or not isinstance(neighbor_count, Integral)
        or neighbor_count < 1
    ):
        raise AdaptiveInputError("neighbor_count must be a positive integer")
    if neighbor_count >= beat_count:
        raise AdaptiveInputError(
            "neighbor_count must be smaller than the beat count"
        )
    return int(neighbor_count)


def _adaptive_weights(distances: np.ndarray) -> np.ndarray:
    positive = distances[distances > 0.0]
    if positive.size == 0:
        return np.ones(distances.size, dtype=np.float64)
    scale = float(np.median(positive))
    return np.exp(-0.5 * (distances / scale) ** 2)


def _cosine_taper(sample_count: int) -> np.ndarray:
    taper = np.ones(sample_count, dtype=np.float64)
    edge_samples = max(2, round(0.05 * sample_count))
    edge_samples = min(edge_samples, sample_count // 2)
    ramp = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, edge_samples))
    taper[:edge_samples] = ramp
    taper[-edge_samples:] = ramp[::-1]
    return taper


def _root_mean_square(values: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(values**2, axis=-1))


def _held_out_rms(epochs: np.ndarray) -> np.ndarray:
    if epochs.shape[0] < 4:
        raise AdaptiveInputError(
            "held-out residual requires at least four complete beats"
        )
    even = epochs[::2]
    odd = epochs[1::2]
    even_template = even.mean(axis=0, keepdims=True)
    odd_template = odd.mean(axis=0, keepdims=True)
    residuals = np.concatenate(
        (even - odd_template, odd - even_template),
        axis=0,
    )
    return np.sqrt(np.mean(residuals**2, axis=(0, 2)))


def _band_power(
    epochs: np.ndarray,
    sampling_rate: float,
    band_hz: tuple[float, float],
) -> float:
    channels = epochs.shape[1]
    continuous = epochs.transpose(1, 0, 2).reshape(channels, -1)
    frequencies, power = welch(
        continuous,
        fs=sampling_rate,
        nperseg=min(epochs.shape[2], 1024),
        axis=-1,
    )
    inside = (frequencies >= band_hz[0]) & (frequencies <= band_hz[1])
    return float(np.median(power[:, inside].sum(axis=1)))


def _validate_recording(eeg: npt.ArrayLike) -> np.ndarray:
    recording = np.asarray(eeg)
    if recording.ndim != 2 or min(recording.shape) < 1:
        raise AdaptiveInputError("eeg must have shape (channels, samples)")
    if np.issubdtype(recording.dtype, np.bool_) or not np.issubdtype(
        recording.dtype,
        np.number,
    ):
        raise AdaptiveInputError("eeg must contain finite numbers")
    recording = recording.astype(np.float64, copy=False)
    if not np.all(np.isfinite(recording)):
        raise AdaptiveInputError("eeg must contain finite numbers")
    return recording


def _validate_ecg(ecg: npt.ArrayLike, sample_count: int) -> np.ndarray:
    reference = np.asarray(ecg)
    if reference.ndim != 1 or reference.size != sample_count:
        raise AdaptiveInputError("ecg must contain one sample per EEG column")
    if np.issubdtype(reference.dtype, np.bool_) or not np.issubdtype(
        reference.dtype,
        np.number,
    ):
        raise AdaptiveInputError("ecg must contain finite numbers")
    reference = reference.astype(np.float64, copy=False)
    if not np.all(np.isfinite(reference)):
        raise AdaptiveInputError("ecg must contain finite numbers")
    return reference


def _validate_peaks(peak_samples: npt.ArrayLike, sample_count: int) -> np.ndarray:
    peaks = np.asarray(peak_samples)
    if peaks.ndim != 1 or peaks.size < 3:
        raise AdaptiveInputError("peak_samples must contain at least three beats")
    if np.issubdtype(peaks.dtype, np.bool_) or not np.issubdtype(
        peaks.dtype,
        np.integer,
    ):
        raise AdaptiveInputError("peak_samples must contain integer samples")
    peaks = peaks.astype(np.int64, copy=False)
    if (
        np.any(peaks < 0)
        or np.any(peaks >= sample_count)
        or np.any(np.diff(peaks) <= 0)
    ):
        raise AdaptiveInputError(
            "peak_samples must be strictly increasing inside the recording"
        )
    return peaks


def _validate_sampling_rate(sampling_rate_hz: float) -> float:
    _validate_finite_number(sampling_rate_hz, name="sampling_rate_hz")
    if sampling_rate_hz <= 0.0:
        raise AdaptiveInputError("sampling_rate_hz must be positive")
    return float(sampling_rate_hz)


def _window_offsets(
    window_seconds: tuple[float, float],
    sampling_rate: float,
    *,
    delay_seconds: float,
) -> tuple[int, int]:
    # Rounded in two steps, as ``bcg.correct_bcg`` does, so both entry points
    # land every window on the same sample.
    delay_samples = round(delay_seconds * sampling_rate)
    start = delay_samples + round(window_seconds[0] * sampling_rate)
    stop = delay_samples + round(window_seconds[1] * sampling_rate)
    if stop <= start:
        raise AdaptiveInputError("a configured window is shorter than one sample")
    return start, stop


def _complete_indices(
    peaks: np.ndarray,
    *,
    sample_count: int,
    windows: tuple[tuple[int, int], ...],
) -> np.ndarray:
    complete = np.ones(peaks.size, dtype=bool)
    for start, stop in windows:
        complete &= peaks + start >= 0
        complete &= peaks + stop <= sample_count
    indices = np.flatnonzero(complete).astype(np.int64, copy=False)
    if indices.size < 3:
        raise AdaptiveInputError("fewer than three complete beat windows remain")
    return indices


def _bandpass(
    signal: np.ndarray,
    sampling_rate: float,
    band_hz: tuple[float, float],
) -> np.ndarray:
    sos = butter(4, band_hz, btype="bandpass", fs=sampling_rate, output="sos")
    try:
        return sosfiltfilt(sos, signal)
    except ValueError as error:
        raise AdaptiveInputError("ecg is too short for morphology filtering") from error


def _morphology_scores(
    epochs: np.ndarray,
    *,
    component_count: int,
    output_samples: int,
) -> np.ndarray:
    source_positions = np.linspace(0.0, 1.0, epochs.shape[1])
    target_positions = np.linspace(0.0, 1.0, output_samples)
    sampled = np.stack(
        [
            np.interp(target_positions, source_positions, epoch)
            for epoch in epochs
        ],
        axis=0,
    )
    sampled -= sampled.mean(axis=1, keepdims=True)
    sampled -= sampled.mean(axis=0, keepdims=True)
    left, singular_values, _ = np.linalg.svd(sampled, full_matrices=False)
    return left[:, :component_count] * singular_values[:component_count]


def _rhythm_features(
    peaks: np.ndarray,
    complete_indices: np.ndarray,
    sampling_rate: float,
) -> np.ndarray:
    intervals = np.diff(peaks) / sampling_rate
    previous = np.concatenate(([intervals[0]], intervals))
    following = np.concatenate((intervals, [intervals[-1]]))
    rhythm = np.column_stack((previous, following, following - previous))
    return rhythm[complete_indices]
