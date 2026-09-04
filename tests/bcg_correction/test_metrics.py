import numpy as np
import pytest

from bcg_correction.metrics import (
    IncrementalTransfer,
    MetricInputError,
    cardiac_locked_rms,
    cardiac_residual_ratio,
    circular_shifted_cardiac_null,
    delay_estimation_eeg,
    held_out_cardiac_rms,
    incremental_signal_transfer,
    is_posterior_eeg_channel,
    regress_out_reference,
    scan_bcg_delay,
    tone_transfer,
    trigger_locked_rms,
    trigger_locked_template,
)

SAMPLING_RATE = 1_000.0
VOLUME_SAMPLES = 900
SAMPLE_COUNT = 22 * VOLUME_SAMPLES
TR_HARMONIC = SAMPLING_RATE / VOLUME_SAMPLES


def test_incremental_transfer_separates_gain_from_distortion() -> None:
    injected = np.array([[1.0, -1.0, 2.0, -2.0]])

    preserved = incremental_signal_transfer(injected, injected)
    attenuated = incremental_signal_transfer(injected, 0.6 * injected)

    assert isinstance(preserved, IncrementalTransfer)
    assert preserved.gain == pytest.approx(1.0)
    assert preserved.relative_error == pytest.approx(0.0)
    assert preserved.cosine_similarity == pytest.approx(1.0)
    assert attenuated.gain == pytest.approx(0.6)
    assert attenuated.relative_error == pytest.approx(0.4)
    assert attenuated.cosine_similarity == pytest.approx(1.0)


def test_reference_regression_is_warning_free_at_microvolt_scale(
    recwarn: pytest.WarningsRecorder,
) -> None:
    samples = np.linspace(0.0, 2.0 * np.pi, 1_000, endpoint=False)
    reference = 1e-3 * np.cos(samples)
    data = (1e-4 * np.sin(samples))[np.newaxis, :]

    residual = regress_out_reference(data, reference)

    assert not recwarn.list
    assert np.all(np.isfinite(residual))


def make_tone(frequency: float, amplitude: float, phase: float) -> np.ndarray:
    times = np.arange(SAMPLE_COUNT, dtype=np.float64) / SAMPLING_RATE
    return amplitude * np.sin(2 * np.pi * frequency * times + phase)


def test_tone_transfer_reports_unit_ratio_for_an_untouched_tone() -> None:
    injected = make_tone(TR_HARMONIC, 5.0, 0.7)

    transfer = tone_transfer(
        injected,
        injected,
        frequency=TR_HARMONIC,
        sampling_rate=SAMPLING_RATE,
    )

    assert transfer.amplitude_ratio == pytest.approx(1.0, abs=1e-9)
    assert transfer.phase_error_degrees == pytest.approx(0.0, abs=1e-9)


def test_tone_transfer_measures_attenuation_and_phase_shift() -> None:
    injected = make_tone(TR_HARMONIC, 5.0, 0.0)
    corrected = make_tone(TR_HARMONIC, 3.0, np.deg2rad(12.0))

    transfer = tone_transfer(
        injected,
        corrected,
        frequency=TR_HARMONIC,
        sampling_rate=SAMPLING_RATE,
    )

    assert transfer.amplitude_ratio == pytest.approx(0.6, abs=1e-6)
    assert transfer.phase_error_degrees == pytest.approx(12.0, abs=1e-6)


def test_tone_transfer_ignores_energy_at_other_frequencies() -> None:
    injected = make_tone(TR_HARMONIC, 5.0, 0.0)
    corrected = injected + make_tone(5 * TR_HARMONIC, 40.0, 1.0)

    transfer = tone_transfer(
        injected,
        corrected,
        frequency=TR_HARMONIC,
        sampling_rate=SAMPLING_RATE,
    )

    assert transfer.amplitude_ratio == pytest.approx(1.0, abs=1e-9)


def test_tone_transfer_wraps_phase_error_to_the_shortest_rotation() -> None:
    injected = make_tone(TR_HARMONIC, 5.0, 0.7)
    corrected = make_tone(TR_HARMONIC, 5.0, 0.7 + np.deg2rad(-170.0))

    transfer = tone_transfer(
        injected,
        corrected,
        frequency=TR_HARMONIC,
        sampling_rate=SAMPLING_RATE,
    )

    assert transfer.phase_error_degrees == pytest.approx(-170.0, abs=1e-6)


def test_trigger_locked_rms_recovers_a_repeating_artifact_amplitude() -> None:
    triggers = np.arange(98) * 200.0 + 50.0
    data = np.zeros((2, SAMPLE_COUNT))
    epoch = 3.0 * np.sin(2 * np.pi * np.arange(200) / 200.0)
    for start in triggers.astype(int):
        data[:, start : start + 200] = np.stack([epoch, -2.0 * epoch])

    locked = trigger_locked_rms(data, triggers, epoch_samples=200)

    expected = np.sqrt(np.mean(epoch**2))
    np.testing.assert_allclose(locked, [expected, 2 * expected], rtol=1e-9)


def test_trigger_locked_template_returns_the_fractional_aligned_average() -> None:
    signal = np.arange(200, dtype=np.float64)
    data = signal[np.newaxis, :]

    template = trigger_locked_template(
        data,
        np.array([20.5, 60.5]),
        epoch_samples=10,
    )

    np.testing.assert_allclose(template[0], np.arange(40.5, 50.5))


def test_trigger_locked_rms_averages_away_signal_unrelated_to_triggers() -> None:
    triggers = np.arange(98) * 200.0 + 50.0
    noise = np.random.default_rng(0).normal(size=(2, SAMPLE_COUNT))

    locked = trigger_locked_rms(noise, triggers, epoch_samples=200)

    assert np.all(locked < 0.2)


def test_held_out_cardiac_rms_uses_opposite_beat_templates() -> None:
    peaks = np.arange(4, dtype=np.int64) * 200 + 50
    artifact = 2.0 * np.sin(2.0 * np.pi * np.arange(40) / 40.0)
    data = np.zeros((2, 1_000), dtype=np.float64)
    for peak in peaks:
        data[0, peak : peak + artifact.size] = artifact
        data[1, peak : peak + artifact.size] = -artifact

    residual = held_out_cardiac_rms(
        data,
        peaks,
        sampling_rate_hz=SAMPLING_RATE,
        window_seconds=(0.0, 0.04),
    )

    np.testing.assert_allclose(residual, 0.0, atol=1e-12)


def test_cardiac_residual_ratio_is_per_channel() -> None:
    peaks = np.arange(4, dtype=np.int64) * 200 + 50
    data = np.zeros((2, 1_000), dtype=np.float64)
    artifact = np.sin(2.0 * np.pi * np.arange(40) / 40.0)
    for index, peak in enumerate(peaks):
        beat = artifact + 0.1 * np.sin(
            2.0 * np.pi * (index + 1) * np.arange(40) / 40.0
        )
        data[0, peak : peak + artifact.size] = beat
        data[1, peak : peak + artifact.size] = 2.0 * beat
    corrected = 0.5 * data

    ratio = cardiac_residual_ratio(
        data,
        corrected,
        peaks,
        sampling_rate_hz=SAMPLING_RATE,
        window_seconds=(0.0, 0.04),
    )

    np.testing.assert_allclose(ratio, [0.5, 0.5], atol=1e-12)


def test_cardiac_locked_rms_is_offset_invariant() -> None:
    peaks = np.array([100, 300, 500, 700], dtype=np.int64)
    data = np.full((2, 1_000), [[100.0], [-250.0]])
    artifact = np.sin(2.0 * np.pi * np.arange(40) / 40.0)
    for peak in peaks:
        data[:, peak : peak + artifact.size] += artifact

    rms = cardiac_locked_rms(
        data,
        peaks,
        sampling_rate_hz=SAMPLING_RATE,
        window_seconds=(0.0, 0.04),
    )

    np.testing.assert_allclose(rms, np.sqrt(0.5), atol=1e-12)


def test_circular_shifted_cardiac_null_is_deterministic_and_unlocked() -> None:
    peaks = np.array([50, 250, 470, 680], dtype=np.int64)
    data = np.zeros((1, 1_000), dtype=np.float64)
    artifact = np.sin(2.0 * np.pi * np.arange(40) / 40.0)
    for peak in peaks:
        data[0, peak : peak + artifact.size] = artifact

    first = circular_shifted_cardiac_null(
        data,
        peaks,
        sampling_rate_hz=SAMPLING_RATE,
        window_seconds=(0.0, 0.04),
        surrogate_count=4,
        seed=20260826,
    )
    second = circular_shifted_cardiac_null(
        data,
        peaks,
        sampling_rate_hz=SAMPLING_RATE,
        window_seconds=(0.0, 0.04),
        surrogate_count=4,
        seed=20260826,
    )

    np.testing.assert_array_equal(first, second)
    assert first.shape == (4, 1)
    assert np.max(first) > 0.1


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: tone_transfer(np.zeros((2, 10)), np.zeros((2, 10)),
                               frequency=1.0, sampling_rate=SAMPLING_RATE),
         "one-dimensional"),
        (lambda: tone_transfer(np.zeros(10), np.zeros(11),
                               frequency=1.0, sampling_rate=SAMPLING_RATE),
         "same length"),
        (lambda: tone_transfer(np.zeros(100), np.zeros(100),
                               frequency=600.0, sampling_rate=SAMPLING_RATE),
         "below the Nyquist"),
        (lambda: trigger_locked_rms(np.zeros((2, 100)), np.array([10.0]),
                                    epoch_samples=0),
         "epoch"),
        (lambda: trigger_locked_rms(np.zeros((2, 100)), np.array([10.0, 95.0]),
                                    epoch_samples=50),
         "beyond the recording"),
        (lambda: held_out_cardiac_rms(
            np.zeros((2, 100)), np.array([10, 20, 30]),
            sampling_rate_hz=SAMPLING_RATE, window_seconds=(0.0, 0.01)),
         "at least four"),
    ],
)
def test_metrics_reject_invalid_inputs(call, message: str) -> None:
    with pytest.raises(MetricInputError, match=message):
        call()


def test_bcg_delay_scan_peaks_at_the_injected_latency() -> None:
    sampling_rate = 1_000.0
    sample_count = 10_000
    peaks = np.array(
        [800, 1_700, 2_600, 3_500, 4_400, 5_300, 6_200, 7_100],
        dtype=np.int64,
    )
    delay_samples = 210
    pulse = np.exp(-0.5 * (np.arange(-30, 31, dtype=np.float64) / 6.0) ** 2)
    data = np.zeros((3, sample_count))
    for peak in peaks:
        centre = int(peak) + delay_samples
        data[0, centre - 30 : centre + 31] += 25.0 * pulse
        data[1, centre - 30 : centre + 31] += 12.0 * pulse
    delays = tuple(index / 100.0 for index in range(0, 41))

    scan = scan_bcg_delay(
        data,
        peaks,
        sampling_rate_hz=sampling_rate,
        delays_seconds=delays,
        window_seconds=(-0.05, 0.05),
    )

    assert scan.best_delay_seconds == pytest.approx(0.21, abs=1e-9)
    assert scan.median_locked_rms[delays.index(0.21)] == pytest.approx(
        max(scan.median_locked_rms)
    )
    assert scan.median_locked_rms[delays.index(0.21)] > 2.0 * scan.median_locked_rms[
        delays.index(0.0)
    ]


def test_posterior_channel_classifier_excludes_frontal_and_ecg() -> None:
    assert is_posterior_eeg_channel("O1")
    assert is_posterior_eeg_channel("POz")
    assert is_posterior_eeg_channel("CP3")
    assert is_posterior_eeg_channel("TP9")
    assert not is_posterior_eeg_channel("Fp1")
    assert not is_posterior_eeg_channel("AF7")
    assert not is_posterior_eeg_channel("Fz")
    assert not is_posterior_eeg_channel("Cz")
    assert not is_posterior_eeg_channel("ECG")


def test_delay_estimation_ignores_qrs_pickup_and_tracks_delayed_bcg() -> None:
    sampling_rate = 1_000.0
    sample_count = 8_000
    peaks = np.array([800, 1_700, 2_600, 3_500, 4_400, 5_300, 6_200], dtype=np.int64)
    qrs = np.exp(-0.5 * (np.arange(-20, 21, dtype=np.float64) / 4.0) ** 2)
    bcg = np.exp(-0.5 * (np.arange(-30, 31, dtype=np.float64) / 6.0) ** 2)
    names = ("Fp1", "Fp2", "Fz", "O1", "ECG")
    data = np.zeros((5, sample_count))
    for peak in peaks:
        data[4, int(peak) - 20 : int(peak) + 21] += qrs
        for frontal in range(3):
            data[frontal, int(peak) - 20 : int(peak) + 21] += 8.0 * qrs
        centre = int(peak) + 210
        data[3, centre - 30 : centre + 31] += 12.0 * bcg
    delays = tuple(index / 100.0 for index in range(0, 41))
    contaminated = scan_bcg_delay(
        data[:4],
        peaks,
        sampling_rate_hz=sampling_rate,
        delays_seconds=delays,
        window_seconds=(-0.05, 0.05),
    )
    prepared = delay_estimation_eeg(data, names, ecg_channel_index=4)
    cleaned = scan_bcg_delay(
        prepared,
        peaks,
        sampling_rate_hz=sampling_rate,
        delays_seconds=delays,
        window_seconds=(-0.05, 0.05),
    )
    assert prepared.shape == (1, sample_count)
    assert contaminated.best_delay_seconds == pytest.approx(0.0, abs=1e-9)
    assert cleaned.best_delay_seconds == pytest.approx(0.21, abs=1e-9)


def test_delay_scan_is_invariant_to_channel_offsets() -> None:
    sampling_rate = 1_000.0
    sample_count = 8_000
    peaks = np.array([800, 1_700, 2_600, 3_500, 4_400, 5_300, 6_200])
    pulse = np.exp(-0.5 * (np.arange(-30, 31, dtype=float) / 6.0) ** 2)
    data = np.zeros((2, sample_count), dtype=float)
    for peak in peaks:
        centre = int(peak) + 210
        data[:, centre - 30 : centre + 31] += pulse
    delays = tuple(index / 100.0 for index in range(0, 41))

    centred = scan_bcg_delay(
        data,
        peaks,
        sampling_rate_hz=sampling_rate,
        delays_seconds=delays,
        window_seconds=(-0.05, 0.05),
    )
    offset = scan_bcg_delay(
        data + np.array([[100.0], [-250.0]]),
        peaks,
        sampling_rate_hz=sampling_rate,
        delays_seconds=delays,
        window_seconds=(-0.05, 0.05),
    )

    assert offset.best_delay_seconds == centred.best_delay_seconds
    np.testing.assert_allclose(
        offset.median_locked_rms,
        centred.median_locked_rms,
        atol=1e-12,
    )
