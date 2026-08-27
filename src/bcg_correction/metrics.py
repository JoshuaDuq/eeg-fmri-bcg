"""Separate artifact-suppression and signal-transfer measurements."""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from numbers import Integral, Real

import numpy as np
import numpy.typing as npt


class MetricInputError(ValueError):
    """Raised when a measurement is asked for something it cannot measure."""


@dataclass(frozen=True, slots=True)
class ToneTransfer:
    """How one exact frequency survived a correction."""

    amplitude_ratio: float
    phase_error_degrees: float


@dataclass(frozen=True, slots=True)
class BcgDelayScan:
    """Trigger-locked BCG energy as a function of ECG-to-BCG delay."""

    delays_seconds: tuple[float, ...]
    median_locked_rms: tuple[float, ...]
    best_delay_seconds: float


RECORDING_DELAY_GRID_SECONDS = tuple(index / 100.0 for index in range(0, 41))
RECORDING_DELAY_WINDOW_SECONDS = (-0.05, 0.05)


def is_posterior_eeg_channel(name: str) -> bool:
    """Return whether a 10-20 label is a posterior EEG site, not ECG or frontal."""
    if not isinstance(name, str) or not name.strip():
        raise MetricInputError("channel names must be nonempty strings")
    token = name.strip().upper().replace(" ", "")
    if token in {"ECG", "EOG", "HEOG", "VEOG"}:
        return False
    if token.startswith(("FP", "AF", "FT", "FC", "F")):
        return False
    return token.startswith(("PO", "CP", "TP", "OZ", "O", "P", "IZ"))


def regress_out_reference(
    data: npt.ArrayLike,
    reference: npt.ArrayLike,
) -> np.ndarray:
    """Remove the contemporaneous linear projection of a reference from each row."""
    recording = _validate_recording(data)
    ref = np.asarray(reference)
    if ref.ndim != 1 or ref.size != recording.shape[1]:
        raise MetricInputError(
            "reference must have one sample per recording column"
        )
    if np.issubdtype(ref.dtype, np.bool_) or not np.issubdtype(ref.dtype, np.number):
        raise MetricInputError("reference must contain finite numeric values")
    ref = ref.astype(np.float64, copy=False)
    if not np.all(np.isfinite(ref)):
        raise MetricInputError("reference must contain finite numeric values")
    centred = ref - np.median(ref)
    energy = float(np.dot(centred, centred))
    if energy == 0.0:
        return recording.copy()
    row_centre = recording - np.median(recording, axis=1, keepdims=True)
    scales = (row_centre @ centred) / energy
    return recording - scales[:, np.newaxis] * centred


def delay_estimation_eeg(
    data: npt.ArrayLike,
    channel_names: Sequence[str],
    *,
    ecg_channel_index: int,
) -> np.ndarray:
    """Return ECG-regressed posterior EEG used to estimate BCG delay."""
    recording = _validate_recording(data)
    names = tuple(channel_names)
    if len(names) != recording.shape[0]:
        raise MetricInputError("channel_names must contain one name per channel")
    if (
        isinstance(ecg_channel_index, bool)
        or not isinstance(ecg_channel_index, Integral)
        or ecg_channel_index < 0
        or ecg_channel_index >= recording.shape[0]
    ):
        raise MetricInputError("ecg_channel_index is outside the recording")
    posterior = [
        index
        for index, name in enumerate(names)
        if index != ecg_channel_index and is_posterior_eeg_channel(name)
    ]
    if not posterior:
        raise MetricInputError("no posterior EEG channels are available")
    return regress_out_reference(
        recording[np.asarray(posterior, dtype=np.int64)],
        recording[int(ecg_channel_index)],
    )


def tone_transfer(
    injected: npt.ArrayLike,
    corrected: npt.ArrayLike,
    *,
    frequency: float,
    sampling_rate: float,
) -> ToneTransfer:
    """Measure amplitude and phase transfer at one exact frequency.

    Both signals are projected onto the same sine/cosine pair. Separation from
    other frequencies is exact when the window holds a whole number of cycles of
    both, which is why benchmarks measure over whole volumes; otherwise strong
    neighbouring energy leaks into the estimate.
    """
    reference = _validate_signal(injected, name="injected")
    result = _validate_signal(corrected, name="corrected")
    if reference.size != result.size:
        raise MetricInputError("injected and corrected must have the same length")
    _validate_frequency(frequency, sampling_rate)

    basis = _tone_basis(reference.size, frequency, sampling_rate)
    reference_phasor = _project(basis, reference)
    result_phasor = _project(basis, result)
    if reference_phasor == 0.0:
        raise MetricInputError("the injected signal has no energy at this frequency")

    rotation = result_phasor / reference_phasor
    return ToneTransfer(
        amplitude_ratio=abs(rotation),
        phase_error_degrees=math.degrees(np.angle(rotation)),
    )


def band_rms_ratio(
    injected: npt.ArrayLike,
    corrected: npt.ArrayLike,
    *,
    low: float,
    high: float,
    sampling_rate: float,
) -> float:
    """Measure the RMS a correction kept inside one frequency band."""
    reference = _validate_signal(injected, name="injected")
    result = _validate_signal(corrected, name="corrected")
    if reference.size != result.size:
        raise MetricInputError("injected and corrected must have the same length")
    _validate_band(low, high, sampling_rate)

    frequencies = np.fft.rfftfreq(reference.size, d=1.0 / sampling_rate)
    inside = (frequencies >= low) & (frequencies <= high)
    reference_power = _band_power(reference, inside)
    if reference_power == 0.0:
        raise MetricInputError("the injected signal has no energy in this band")
    return math.sqrt(_band_power(result, inside) / reference_power)


def trigger_locked_rms(
    data: npt.ArrayLike,
    triggers: npt.ArrayLike,
    *,
    epoch_samples: int,
) -> npt.NDArray[np.float64]:
    """Measure the per-channel RMS of the artifact that repeats at the triggers.

    Anything not locked to the triggers averages towards zero, so this isolates
    residual artifact from ongoing EEG. Fractional trigger positions are sampled
    by linear interpolation rather than rounded to a different timing grid.
    """
    recording = _validate_recording(data)
    template = trigger_locked_template(
        recording,
        triggers,
        epoch_samples=epoch_samples,
    )
    return np.sqrt(np.mean(template**2, axis=1))


def trigger_locked_template(
    data: npt.ArrayLike,
    triggers: npt.ArrayLike,
    *,
    epoch_samples: int,
) -> npt.NDArray[np.float64]:
    """Return the fractional-position average of epochs at ``triggers``."""
    recording = _validate_recording(data)
    positions = _validate_triggers(
        triggers,
        epoch_samples=epoch_samples,
        sample_count=recording.shape[1],
    )
    epochs = _extract_fractional_epochs(recording, positions, epoch_samples)
    return epochs.mean(axis=1)


def scan_bcg_delay(
    data: npt.ArrayLike,
    peak_samples: npt.ArrayLike,
    *,
    sampling_rate_hz: float,
    delays_seconds: tuple[float, ...],
    window_seconds: tuple[float, float],
) -> BcgDelayScan:
    """Score ECG-to-BCG delays by median heartbeat-locked EEG RMS.

    Each delay is applied to the R samples, then a short epoch around the
    delayed anchor is averaged across beats. Unlocked EEG averages down, so
    the delay with the largest median RMS is the BCG occurrence.
    Incomplete edge epochs are dropped per delay rather than aborting the scan.
    """
    recording = _validate_recording(data)
    recording = recording - np.median(recording, axis=1, keepdims=True)
    sampling_rate = _validate_sampling_rate(sampling_rate_hz)
    delays = _validate_delay_grid(delays_seconds)
    rel_start, epoch_samples = _validate_delay_window(window_seconds, sampling_rate)
    peaks = _validate_delay_peaks(peak_samples, recording.shape[1])

    scores: list[float] = []
    for delay in delays:
        starts = peaks.astype(np.float64) + round(delay * sampling_rate) + rel_start
        keep = (starts >= 0.0) & (
            starts + epoch_samples - 1.0 <= recording.shape[1] - 1
        )
        kept = starts[keep]
        if kept.size < 2:
            scores.append(float("nan"))
            continue
        template = trigger_locked_template(
            recording,
            kept,
            epoch_samples=epoch_samples,
        )
        centre = -rel_start
        if centre < 0 or centre >= epoch_samples:
            scores.append(float("nan"))
            continue
        radius = min(5, centre, epoch_samples - centre - 1)
        core = template[:, centre - radius : centre + radius + 1]
        scores.append(float(np.median(np.sqrt(np.mean(core**2, axis=1)))))

    finite = [index for index, score in enumerate(scores) if math.isfinite(score)]
    if not finite:
        raise MetricInputError("no delay retained two complete BCG epochs")
    best_index = max(finite, key=lambda index: scores[index])
    return BcgDelayScan(
        delays_seconds=delays,
        median_locked_rms=tuple(scores),
        best_delay_seconds=delays[best_index],
    )


def estimate_ecg_to_bcg_delay(
    eeg: npt.ArrayLike,
    peak_samples: npt.ArrayLike,
    *,
    sampling_rate_hz: float,
) -> BcgDelayScan:
    """Estimate one recording-level ECG-to-BCG delay from EEG and R samples."""
    return scan_bcg_delay(
        eeg,
        peak_samples,
        sampling_rate_hz=sampling_rate_hz,
        delays_seconds=RECORDING_DELAY_GRID_SECONDS,
        window_seconds=RECORDING_DELAY_WINDOW_SECONDS,
    )


def event_locked_rms_ratio(
    injected: npt.ArrayLike,
    corrected: npt.ArrayLike,
    event_starts: npt.ArrayLike,
    *,
    epoch_samples: int,
) -> float:
    """Measure event-locked RMS transfer using fractional event positions."""
    reference = _validate_signal(injected, name="injected")
    result = _validate_signal(corrected, name="corrected")
    if reference.size != result.size:
        raise MetricInputError("injected and corrected must have the same length")
    positions = _validate_triggers(
        event_starts,
        epoch_samples=epoch_samples,
        sample_count=reference.size,
    )
    reference_epochs = _extract_fractional_epochs(
        reference[np.newaxis, :],
        positions,
        epoch_samples,
    )[0]
    result_epochs = _extract_fractional_epochs(
        result[np.newaxis, :],
        positions,
        epoch_samples,
    )[0]
    reference_rms = np.sqrt(np.mean(reference_epochs.mean(axis=0) ** 2))
    if reference_rms == 0.0:
        raise MetricInputError("the injected event has no event-locked energy")
    return float(np.sqrt(np.mean(result_epochs.mean(axis=0) ** 2)) / reference_rms)


def held_out_cardiac_rms(
    data_uv: npt.ArrayLike,
    peak_samples: npt.ArrayLike,
    *,
    sampling_rate_hz: float,
    window_seconds: tuple[float, float],
) -> npt.NDArray[np.float64]:
    """Measure leakage-controlled, held-out cardiac residual RMS per channel.

    Even-indexed beats are scored against the odd-indexed template and vice
    versa. Consequently, an event never contributes to the template used to
    score that same event.
    """
    recording = _validate_recording(data_uv)
    epochs = _cardiac_epochs(
        recording,
        peak_samples,
        sampling_rate_hz=sampling_rate_hz,
        window_seconds=window_seconds,
    )
    if epochs.shape[1] < 4:
        raise MetricInputError(
            "held-out cardiac RMS requires at least four complete beats"
        )
    even_epochs = epochs[:, ::2]
    odd_epochs = epochs[:, 1::2]
    even_template = even_epochs.mean(axis=1, keepdims=True)
    odd_template = odd_epochs.mean(axis=1, keepdims=True)
    residuals = np.concatenate(
        (even_epochs - odd_template, odd_epochs - even_template),
        axis=1,
    )
    return np.sqrt(np.mean(residuals**2, axis=(1, 2)))


def cardiac_residual_ratio(
    before_uv: npt.ArrayLike,
    after_uv: npt.ArrayLike,
    peak_samples: npt.ArrayLike,
    *,
    sampling_rate_hz: float,
    window_seconds: tuple[float, float],
) -> npt.NDArray[np.float64]:
    """Return held-out cardiac residual ratios for each channel."""
    before = _validate_recording(before_uv)
    after = _validate_recording(after_uv)
    if before.shape != after.shape:
        raise MetricInputError("before and after must have the same shape")
    before_rms = held_out_cardiac_rms(
        before,
        peak_samples,
        sampling_rate_hz=sampling_rate_hz,
        window_seconds=window_seconds,
    )
    after_rms = held_out_cardiac_rms(
        after,
        peak_samples,
        sampling_rate_hz=sampling_rate_hz,
        window_seconds=window_seconds,
    )
    if np.any(before_rms == 0.0):
        raise MetricInputError(
            "before contains a channel with no held-out cardiac residual"
        )
    return after_rms / before_rms


def cardiac_locked_rms(
    data: npt.ArrayLike,
    peak_samples: npt.ArrayLike,
    *,
    sampling_rate_hz: float,
    window_seconds: tuple[float, float],
) -> npt.NDArray[np.float64]:
    """Measure channel-centred, heartbeat-locked RMS per channel.

    Only complete epochs are retained. Removing each channel's recording-level
    median prevents amplifier offsets from being counted as cardiac energy.
    """
    recording = _validate_recording(data)
    sampling_rate = _validate_sampling_rate(sampling_rate_hz)
    rel_start, epoch_samples = _validate_delay_window(
        window_seconds,
        sampling_rate,
    )
    peaks = _validate_delay_peaks(peak_samples, recording.shape[1])
    starts = peaks + rel_start
    complete = (starts >= 0) & (
        starts + epoch_samples <= recording.shape[1]
    )
    starts = starts[complete]
    if starts.size < 2:
        raise MetricInputError(
            "cardiac-locked RMS requires at least two complete beats"
        )
    centred = recording - np.median(recording, axis=1, keepdims=True)
    return trigger_locked_rms(
        centred,
        starts.astype(np.float64),
        epoch_samples=epoch_samples,
    )


def circular_shifted_cardiac_null(
    data_uv: npt.ArrayLike,
    peak_samples: npt.ArrayLike,
    *,
    sampling_rate_hz: float,
    window_seconds: tuple[float, float],
    surrogate_count: int,
    seed: int,
) -> npt.NDArray[np.float64]:
    """Generate deterministic circular-shift cardiac residual nulls.

    Each surrogate keeps the supplied event intervals unchanged while
    shifting the recording phase relative to those events. A zero shift is
    excluded so the observed alignment cannot be returned as a null sample.
    """
    recording = _validate_recording(data_uv)
    _validate_cardiac_epoch_positions(
        recording,
        peak_samples,
        sampling_rate_hz=sampling_rate_hz,
        window_seconds=window_seconds,
    )
    if (
        isinstance(surrogate_count, bool)
        or not isinstance(surrogate_count, int)
        or surrogate_count < 1
    ):
        raise MetricInputError("surrogate_count must be a positive integer")
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed < 0
    ):
        raise MetricInputError("seed must be a nonnegative integer")
    generator = np.random.default_rng(seed)
    null_values = np.empty(
        (surrogate_count, recording.shape[0]),
        dtype=np.float64,
    )
    for surrogate_index in range(surrogate_count):
        shift = int(generator.integers(1, recording.shape[1]))
        shifted = np.roll(recording, shift, axis=1)
        null_values[surrogate_index] = held_out_cardiac_rms(
            shifted,
            peak_samples,
            sampling_rate_hz=sampling_rate_hz,
            window_seconds=window_seconds,
        )
    return null_values


def _validate_signal(signal: npt.ArrayLike, *, name: str) -> np.ndarray:
    values = np.asarray(signal)
    if values.ndim != 1 or values.size < 2:
        raise MetricInputError(f"{name} must be a one-dimensional signal")
    if np.issubdtype(values.dtype, np.bool_) or not np.issubdtype(
        values.dtype, np.number
    ):
        raise MetricInputError(f"{name} must contain finite numbers")
    values = values.astype(np.float64, copy=False)
    if not np.all(np.isfinite(values)):
        raise MetricInputError(f"{name} must contain finite numbers")
    return values


def _validate_recording(data: npt.ArrayLike) -> np.ndarray:
    recording = np.asarray(data)
    if recording.ndim != 2 or recording.shape[0] == 0 or recording.shape[1] == 0:
        raise MetricInputError("data must have shape (channels, samples)")
    if np.issubdtype(recording.dtype, np.bool_) or not np.issubdtype(
        recording.dtype, np.number
    ):
        raise MetricInputError("data must contain finite numbers")
    if not np.all(np.isfinite(recording)):
        raise MetricInputError("data must contain finite numbers")
    return recording.astype(np.float64, copy=False)


def _validate_frequency(frequency: float, sampling_rate: float) -> None:
    if (
        isinstance(frequency, bool)
        or not isinstance(frequency, Real)
        or not math.isfinite(float(frequency))
        or frequency <= 0.0
    ):
        raise MetricInputError("frequency must be finite and positive")
    if (
        isinstance(sampling_rate, bool)
        or not isinstance(sampling_rate, Real)
        or not math.isfinite(float(sampling_rate))
        or sampling_rate <= 0.0
    ):
        raise MetricInputError("sampling rate must be finite and positive")
    if frequency >= 0.5 * sampling_rate:
        raise MetricInputError("frequency must stay below the Nyquist frequency")


def _validate_band(low: float, high: float, sampling_rate: float) -> None:
    _validate_frequency(high, sampling_rate)
    if (
        isinstance(low, bool)
        or not isinstance(low, Real)
        or not math.isfinite(float(low))
        or low < 0.0
        or low >= high
    ):
        raise MetricInputError("the band must be positive and increasing")


def _validate_triggers(
    triggers: npt.ArrayLike,
    *,
    epoch_samples: int,
    sample_count: int,
) -> np.ndarray:
    if not isinstance(epoch_samples, int) or epoch_samples < 1:
        raise MetricInputError("epoch samples must be a positive integer")
    positions = np.asarray(triggers)
    if positions.ndim != 1 or positions.size == 0:
        raise MetricInputError("triggers must be a nonempty one-dimensional array")
    if np.issubdtype(positions.dtype, np.bool_) or not np.issubdtype(
        positions.dtype, np.number
    ):
        raise MetricInputError("triggers must contain finite numbers")
    positions = positions.astype(np.float64, copy=False)
    if not np.all(np.isfinite(positions)) or np.any(positions < 0.0):
        raise MetricInputError("triggers must contain finite nonnegative numbers")
    if np.any(np.diff(positions) <= 0.0):
        raise MetricInputError("triggers must be strictly increasing")
    if positions[-1] + epoch_samples - 1.0 > sample_count - 1:
        raise MetricInputError("the measured epochs extend beyond the recording")
    return positions


def _cardiac_epochs(
    recording: np.ndarray,
    peak_samples: npt.ArrayLike,
    *,
    sampling_rate_hz: float,
    window_seconds: tuple[float, float],
) -> np.ndarray:
    positions, epoch_samples = _validate_cardiac_epoch_positions(
        recording,
        peak_samples,
        sampling_rate_hz=sampling_rate_hz,
        window_seconds=window_seconds,
    )
    return _extract_fractional_epochs(recording, positions, epoch_samples)


def _validate_cardiac_epoch_positions(
    recording: np.ndarray,
    peak_samples: npt.ArrayLike,
    *,
    sampling_rate_hz: float,
    window_seconds: tuple[float, float],
) -> tuple[np.ndarray, int]:
    _validate_sampling_rate(sampling_rate_hz)
    if (
        not isinstance(window_seconds, tuple)
        or len(window_seconds) != 2
        or not all(
            isinstance(value, Real)
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in window_seconds
        )
        or window_seconds[0] >= window_seconds[1]
    ):
        raise MetricInputError(
            "window_seconds must be a finite increasing pair"
        )
    peaks = np.asarray(peak_samples)
    if peaks.ndim != 1 or peaks.size == 0:
        raise MetricInputError(
            "peak_samples must be a nonempty one-dimensional array"
        )
    if np.issubdtype(peaks.dtype, np.bool_) or not np.issubdtype(
        peaks.dtype,
        np.integer,
    ):
        raise MetricInputError("peak_samples must contain integer samples")
    peaks = peaks.astype(np.int64, copy=False)
    if np.any(peaks < 0) or np.any(peaks >= recording.shape[1]):
        raise MetricInputError(
            "peak_samples must stay inside the recording"
        )
    if np.any(np.diff(peaks) <= 0):
        raise MetricInputError("peak_samples must be strictly increasing")
    window_start = round(window_seconds[0] * sampling_rate_hz)
    window_stop = round(window_seconds[1] * sampling_rate_hz)
    epoch_samples = window_stop - window_start
    if epoch_samples < 1:
        raise MetricInputError("window_seconds is shorter than one sample")
    positions = peaks + window_start
    if positions[0] < 0 or positions[-1] + epoch_samples > recording.shape[1]:
        raise MetricInputError(
            "the measured cardiac epochs extend beyond the recording"
        )
    return positions.astype(np.float64), epoch_samples


def _validate_delay_grid(delays_seconds: tuple[float, ...]) -> tuple[float, ...]:
    if not isinstance(delays_seconds, tuple) or not delays_seconds:
        raise MetricInputError("delays_seconds must be a nonempty tuple")
    delays = []
    for delay in delays_seconds:
        if (
            isinstance(delay, bool)
            or not isinstance(delay, Real)
            or not math.isfinite(float(delay))
            or float(delay) < 0.0
        ):
            raise MetricInputError(
                "delays_seconds must contain finite nonnegative numbers"
            )
        delays.append(float(delay))
    if any(left >= right for left, right in pairwise(delays)):
        raise MetricInputError("delays_seconds must be strictly increasing")
    return tuple(delays)


def _validate_delay_window(
    window_seconds: tuple[float, float],
    sampling_rate: float,
) -> tuple[int, int]:
    if (
        not isinstance(window_seconds, tuple)
        or len(window_seconds) != 2
        or not all(
            isinstance(value, Real)
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in window_seconds
        )
        or window_seconds[0] >= window_seconds[1]
    ):
        raise MetricInputError("window_seconds must be a finite increasing pair")
    rel_start = round(window_seconds[0] * sampling_rate)
    epoch_samples = round(window_seconds[1] * sampling_rate) - rel_start
    if epoch_samples < 1:
        raise MetricInputError("window_seconds is shorter than one sample")
    return rel_start, epoch_samples


def _validate_delay_peaks(
    peak_samples: npt.ArrayLike,
    sample_count: int,
) -> np.ndarray:
    peaks = np.asarray(peak_samples)
    if peaks.ndim != 1 or peaks.size < 2:
        raise MetricInputError("peak_samples must contain at least two events")
    if np.issubdtype(peaks.dtype, np.bool_) or not np.issubdtype(
        peaks.dtype,
        np.integer,
    ):
        raise MetricInputError("peak_samples must contain integer samples")
    peaks = peaks.astype(np.int64, copy=False)
    if np.any(peaks < 0) or np.any(peaks >= sample_count):
        raise MetricInputError("peak_samples must stay inside the recording")
    if np.any(np.diff(peaks) <= 0):
        raise MetricInputError("peak_samples must be strictly increasing")
    return peaks


def _validate_sampling_rate(sampling_rate_hz: float) -> float:
    if (
        isinstance(sampling_rate_hz, bool)
        or not isinstance(sampling_rate_hz, Real)
        or not math.isfinite(float(sampling_rate_hz))
        or sampling_rate_hz <= 0.0
    ):
        raise MetricInputError("sampling rate must be finite and positive")
    return float(sampling_rate_hz)


def _extract_fractional_epochs(
    recording: np.ndarray,
    positions: np.ndarray,
    epoch_samples: int,
) -> np.ndarray:
    offsets = np.arange(epoch_samples, dtype=np.float64)
    sample_positions = positions[:, np.newaxis] + offsets
    lower = np.floor(sample_positions).astype(np.int64)
    upper = np.minimum(lower + 1, recording.shape[1] - 1)
    fraction = sample_positions - lower
    return (
        recording[:, lower] * (1.0 - fraction[np.newaxis, :, :])
        + recording[:, upper] * fraction[np.newaxis, :, :]
    )


def _tone_basis(
    sample_count: int,
    frequency: float,
    sampling_rate: float,
) -> np.ndarray:
    times = np.arange(sample_count, dtype=np.float64) / sampling_rate
    return np.stack(
        [np.sin(2 * np.pi * frequency * times), np.cos(2 * np.pi * frequency * times)],
        axis=1,
    )


def _project(basis: np.ndarray, signal: np.ndarray) -> complex:
    coefficients, *_ = np.linalg.lstsq(basis, signal, rcond=None)
    return complex(coefficients[0], coefficients[1])


def _band_power(signal: np.ndarray, inside: np.ndarray) -> float:
    spectrum = np.fft.rfft(signal)
    return float(np.sum(np.abs(spectrum[inside]) ** 2))
