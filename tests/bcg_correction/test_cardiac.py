from dataclasses import replace

import numpy as np
import pytest

from bcg_correction.bcg_config import DetectorConfig
from bcg_correction.cardiac import (
    CardiacInputError,
    _quality_summary,
    _select_events,
    _select_polarity_seed,
    detect_r_peaks,
)


def make_ecg(
    sampling_rate_hz: float,
    duration_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    samples = np.arange(round(duration_seconds * sampling_rate_hz), dtype=float)
    signal = np.zeros(samples.size, dtype=float)
    peak_seconds = np.array([0.8, 1.65, 2.53, 3.44, 4.37, 5.31])
    for peak_second in peak_seconds:
        centre = peak_second * sampling_rate_hz
        width = 0.008 * sampling_rate_hz
        signal += np.exp(-0.5 * ((samples - centre) / width) ** 2)
        t_wave = centre + 0.28 * sampling_rate_hz
        signal += 0.65 * np.exp(
            -0.5 * ((samples - t_wave) / (0.035 * sampling_rate_hz)) ** 2
        )
    signal += 0.03 * np.sin(2.0 * np.pi * samples / (sampling_rate_hz * 7.0))
    return signal, np.rint(peak_seconds * sampling_rate_hz).astype(np.int64)


@pytest.fixture
def detector_config() -> DetectorConfig:
    return DetectorConfig(
        ecg_channel="ECG",
        preprocessing_band_hz=(7.0, 40.0),
        teager_emphasis_hz=10.0,
        teager_smoothing_seconds=0.028,
        template_window_seconds=(-0.2, 0.4),
        minimum_rr_seconds=0.4,
        maximum_rr_seconds=1.5,
        candidate_refractory_seconds=0.25,
        candidate_prominence_mad=3.0,
        correlation_threshold=0.5,
        refinement_iterations=2,
    )


@pytest.fixture
def biphasic_ecg() -> tuple[np.ndarray, np.ndarray]:
    samples = np.arange(10_000, dtype=float)
    expected = np.arange(600, 9_601, 900, dtype=np.int64)
    signal = np.zeros(samples.size, dtype=float)
    for peak in expected:
        signal += np.exp(-0.5 * ((samples - peak) / 8.0) ** 2)
        secondary = peak + 320
        signal -= 0.8 * np.exp(-0.5 * ((samples - secondary) / 10.0) ** 2)
    return signal, expected


@pytest.fixture
def opposite_polarity_disturbance_ecg() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    int,
]:
    sampling_rate_hz = 1_000.0
    samples = np.arange(6_000, dtype=float)
    expected = np.array([800, 1_650, 2_530, 3_440, 4_370, 5_310])
    undisturbed = 0.03 * np.sin(2.0 * np.pi * samples / (sampling_rate_hz * 7.0))
    for peak in expected:
        undisturbed += np.exp(-0.5 * ((samples - peak) / 8.0) ** 2)
    disturbed_peak = int(expected[4])
    qrs = np.exp(-0.5 * ((samples - disturbed_peak) / 8.0) ** 2)
    disturbed = undisturbed - 2.0 * qrs
    return undisturbed, disturbed, expected, disturbed_peak


def assert_peaks_match(
    detected: np.ndarray,
    expected: np.ndarray,
    *,
    tolerance_samples: int,
) -> None:
    assert detected.shape == expected.shape
    assert np.all(np.abs(detected - expected) <= tolerance_samples)


def test_detector_recovers_weak_qrs_after_established_period(
    detector_config: DetectorConfig,
) -> None:
    sampling_rate_hz = 1_000.0
    samples = np.arange(16_000, dtype=float)
    expected = np.arange(800, 15_200, 900, dtype=np.int64)
    ecg = 0.03 * np.sin(2.0 * np.pi * samples / 7_000.0)
    for index, peak in enumerate(expected):
        weak = 8 <= index <= 10
        amplitude = 0.15 if weak else 1.0
        width = 25.0 if weak else 8.0
        ecg += amplitude * np.exp(-0.5 * ((samples - peak) / width) ** 2)
        ecg += 0.35 * np.exp(-0.5 * ((samples - peak - 320) / 30.0) ** 2)

    detection = detect_r_peaks(ecg, sampling_rate_hz, config=detector_config)

    assert_peaks_match(detection.peak_samples, expected, tolerance_samples=20)
    t_waves = expected[8:11] + 320
    assert not np.any(
        np.min(np.abs(detection.peak_samples[:, None] - t_waves), axis=1) <= 15
    )


def test_detector_does_not_insert_mid_interval_beats_on_slow_rr(
    detector_config: DetectorConfig,
) -> None:
    sampling_rate_hz = 1_000.0
    samples = np.arange(20_000, dtype=float)
    expected = np.arange(800, 19_000, 1_580, dtype=np.int64)
    ecg = 0.03 * np.sin(2.0 * np.pi * samples / 7_000.0)
    for peak in expected:
        ecg += np.exp(-0.5 * ((samples - peak) / 8.0) ** 2)
        ecg += 0.5 * np.exp(-0.5 * ((samples - peak - 320) / 30.0) ** 2)

    detection = detect_r_peaks(ecg, sampling_rate_hz, config=detector_config)

    assert_peaks_match(detection.peak_samples, expected, tolerance_samples=20)


def test_detector_does_not_fill_a_true_pause_with_t_waves(
    detector_config: DetectorConfig,
) -> None:
    sampling_rate_hz = 1_000.0
    samples = np.arange(16_000, dtype=float)
    expected = np.arange(800, 15_200, 900, dtype=np.int64)
    present = []
    ecg = 0.03 * np.sin(2.0 * np.pi * samples / 7_000.0)
    for index, peak in enumerate(expected):
        keep = index < 6 or index % 2 == 0
        t_wave = peak + 320
        ecg += 0.5 * np.exp(-0.5 * ((samples - t_wave) / 30.0) ** 2)
        if keep:
            present.append(int(peak))
            ecg += np.exp(-0.5 * ((samples - peak) / 8.0) ** 2)
    present = np.asarray(present, dtype=np.int64)

    detection = detect_r_peaks(ecg, sampling_rate_hz, config=detector_config)

    t_waves = expected[6::2] + 320
    assert not np.any(
        np.min(np.abs(detection.peak_samples[:, None] - t_waves), axis=1) <= 15
    )
    distances = np.min(np.abs(detection.peak_samples[:, None] - present), axis=1)
    assert np.all(distances <= 20)


def test_detector_recovers_known_qrs_positions(
    detector_config: DetectorConfig,
) -> None:
    ecg, expected = make_ecg(1_000.0, 6.0)

    detection = detect_r_peaks(ecg, 1_000.0, config=detector_config)

    assert_peaks_match(detection.peak_samples, expected, tolerance_samples=10)
    assert np.all(np.diff(detection.peak_samples) > 0)
    assert detection.quality.status == "ok"
    assert detection.quality.candidate_count > detection.quality.accepted_count
    assert detection.quality.rejected_double_mark > 0


def test_detector_is_invariant_to_global_ecg_polarity(
    detector_config: DetectorConfig,
) -> None:
    ecg, _ = make_ecg(1_000.0, 6.0)

    positive = detect_r_peaks(ecg, 1_000.0, config=detector_config)
    negative = detect_r_peaks(-ecg, 1_000.0, config=detector_config)

    assert np.array_equal(positive.peak_samples, negative.peak_samples)
    assert positive.quality.selected_polarity == 1
    assert negative.quality.selected_polarity == -1


def test_detector_uses_one_recording_polarity(
    detector_config: DetectorConfig,
    biphasic_ecg: tuple[np.ndarray, np.ndarray],
) -> None:
    ecg, expected = biphasic_ecg

    detection = detect_r_peaks(ecg, 1_000.0, config=detector_config)

    assert_peaks_match(detection.peak_samples, expected, tolerance_samples=10)
    assert detection.quality.selected_polarity == 1
    assert detection.quality.positive_candidate_count > 0
    assert detection.quality.negative_candidate_count > 0
    assert (
        detection.quality.positive_candidate_count
        + detection.quality.negative_candidate_count
        == detection.quality.candidate_count
    )
    assert detection.quality.degradation_reasons == ()
    assert detection.quality.status == "ok"


def test_polarity_alignment_uses_original_candidate_window(
    detector_config: DetectorConfig,
) -> None:
    sampling_rate_hz = 1_000.0
    candidates = np.array([500, 1_400, 2_300], dtype=np.int64)
    scores = np.ones(candidates.size, dtype=float)
    conditioned = np.zeros(2_500, dtype=float)
    signed_extrema = candidates + 60
    outside_original_window = candidates + 100
    conditioned[signed_extrema] = 2.0
    conditioned[outside_original_window] = 3.0

    seed = _select_polarity_seed(
        candidates,
        scores,
        conditioned,
        sampling_rate=sampling_rate_hz,
        config=detector_config,
    )

    np.testing.assert_array_equal(seed.candidates.peaks, signed_extrema)


def test_polarity_seed_rejects_no_eligible_arm(
    detector_config: DetectorConfig,
) -> None:
    candidates = np.array([500, 1_400], dtype=np.int64)
    conditioned = np.zeros(2_000, dtype=float)
    conditioned[candidates] = 1.0

    with pytest.raises(
        CardiacInputError,
        match=r"^ECG detector found no physiological polarity arm$",
    ):
        _select_polarity_seed(
            candidates,
            np.ones(candidates.size),
            conditioned,
            sampling_rate=1_000.0,
            config=detector_config,
        )


def test_polarity_seed_rejects_exact_score_tie(
    detector_config: DetectorConfig,
) -> None:
    positive = np.array([500, 1_400, 2_300], dtype=np.int64)
    negative = np.array([700, 1_600, 2_500], dtype=np.int64)
    candidates = np.sort(np.concatenate((positive, negative)))
    conditioned = np.zeros(3_000, dtype=float)
    conditioned[positive] = 1.0
    conditioned[negative] = -1.0

    with pytest.raises(
        CardiacInputError,
        match=r"^ECG polarity is ambiguous$",
    ):
        _select_polarity_seed(
            candidates,
            np.ones(candidates.size),
            conditioned,
            sampling_rate=1_000.0,
            config=detector_config,
        )


def test_polarity_seed_scores_arms_before_secondary_rejection(
    detector_config: DetectorConfig,
) -> None:
    positive = np.array([500, 1_400, 2_300, 3_200], dtype=np.int64)
    negative = np.array([700, 1_600, 2_500, 3_400], dtype=np.int64)
    candidates = np.sort(np.concatenate((positive, negative)))
    conditioned = np.zeros(4_000, dtype=float)
    conditioned[positive] = 1.0
    conditioned[negative] = -1.0
    scores = np.ones(candidates.size, dtype=float)
    scores[candidates == positive[0]] = 2.0

    with pytest.raises(
        CardiacInputError,
        match=r"^ECG polarity is ambiguous$",
    ):
        _select_polarity_seed(
            candidates,
            scores,
            conditioned,
            sampling_rate=1_000.0,
            config=detector_config,
        )


def test_opposite_polarity_artifact_cannot_veto_selected_arm_candidate(
    detector_config: DetectorConfig,
) -> None:
    positive = np.array([500, 1_400, 2_300, 3_200, 4_100], dtype=np.int64)
    negative = np.array([800], dtype=np.int64)
    candidates = np.sort(np.concatenate((positive, negative)))
    conditioned = np.zeros(4_500, dtype=float)
    conditioned[positive] = 1.0
    conditioned[negative] = -1.0
    scores = np.ones(candidates.size, dtype=float)
    scores[candidates == negative[0]] = 2.0

    seed = _select_polarity_seed(
        candidates,
        scores,
        conditioned,
        sampling_rate=1_000.0,
        config=detector_config,
    )

    assert seed.selected_polarity == 1
    np.testing.assert_array_equal(seed.candidates.peaks, positive)


def test_opposite_polarity_disturbance_is_not_normalized(
    detector_config: DetectorConfig,
    opposite_polarity_disturbance_ecg: tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        int,
    ],
) -> None:
    sampling_rate_hz = 1_000.0
    undisturbed_ecg, disturbed_ecg, expected, disturbed_peak = (
        opposite_polarity_disturbance_ecg
    )

    undisturbed = detect_r_peaks(
        undisturbed_ecg,
        sampling_rate_hz,
        config=detector_config,
    )
    disturbed = detect_r_peaks(
        disturbed_ecg,
        sampling_rate_hz,
        config=detector_config,
    )

    assert disturbed.quality.selected_polarity == undisturbed.quality.selected_polarity
    assert disturbed.peak_samples.size <= undisturbed.peak_samples.size
    assert not np.any(np.abs(disturbed.peak_samples - disturbed_peak) <= 10)
    distances = np.min(
        np.abs(disturbed.peak_samples[:, None] - expected),
        axis=1,
    )
    assert np.all(distances <= 10)


@pytest.mark.parametrize("disturbed_index", [0, 4, 5])
def test_opposite_polarity_disturbance_does_not_admit_same_cycle_t_wave(
    detector_config: DetectorConfig,
    disturbed_index: int,
) -> None:
    sampling_rate_hz = 1_000.0
    ecg, expected = make_ecg(sampling_rate_hz, 6.0)
    samples = np.arange(ecg.size, dtype=float)
    disturbed_peak = int(expected[disturbed_index])
    qrs = np.exp(-0.5 * ((samples - disturbed_peak) / 8.0) ** 2)
    disturbed_ecg = ecg - 2.0 * qrs

    detection = detect_r_peaks(
        disturbed_ecg,
        sampling_rate_hz,
        config=detector_config,
    )

    assert detection.quality.selected_polarity == 1
    assert detection.peak_samples.size <= expected.size
    assert not np.any(np.abs(detection.peak_samples - disturbed_peak) <= 10)
    t_wave = disturbed_peak + 280
    assert not np.any(np.abs(detection.peak_samples - t_wave) <= 10)
    assert detection.quality.rejected_low_prominence == 1
    assert "low_prominence_candidate" in detection.quality.degradation_reasons
    assert detection.quality.status == "degraded"


def test_detector_preserves_same_polarity_ectopic_timing(
    detector_config: DetectorConfig,
) -> None:
    sampling_rate_hz = 1_000.0
    samples = np.arange(5_000, dtype=float)
    expected = np.array([600, 1_500, 2_100, 3_300, 4_200])
    ecg = 0.001 * np.sin(2.0 * np.pi * samples / 7_000.0)
    for peak in expected:
        amplitude = 0.8 if peak == 2_100 else 1.0
        ecg += amplitude * np.exp(-0.5 * ((samples - peak) / 8.0) ** 2)

    detection = detect_r_peaks(
        ecg,
        sampling_rate_hz,
        config=detector_config,
    )

    np.testing.assert_array_equal(detection.peak_samples, expected)
    assert detection.quality.degradation_reasons == ()
    assert detection.quality.status == "ok"


def test_detector_rejects_t_wave_candidates(
    detector_config: DetectorConfig,
) -> None:
    ecg, expected = make_ecg(1_000.0, 6.0)

    detection = detect_r_peaks(ecg, 1_000.0, config=detector_config)
    t_wave_positions = expected + 280

    assert_peaks_match(detection.peak_samples, expected, tolerance_samples=10)
    assert not np.any(
        np.min(np.abs(detection.peak_samples[:, None] - t_wave_positions), axis=1)
        <= 10
    )


def test_detector_is_deterministic_and_annotation_independent(
    detector_config: DetectorConfig,
) -> None:
    ecg, _ = make_ecg(1_000.0, 6.0)
    external_marker_train = np.array([800, 2530, 3440, 4370, 5310])
    first = detect_r_peaks(ecg, 1_000.0, config=detector_config)
    second = detect_r_peaks(ecg, 1_000.0, config=detector_config)

    assert first.peak_samples.tobytes() == second.peak_samples.tobytes()
    assert first.quality == second.quality
    assert external_marker_train.size < first.peak_samples.size
    assert "annotations" not in detect_r_peaks.__code__.co_varnames


def test_detector_enforces_candidate_refractory_interval(
    detector_config: DetectorConfig,
) -> None:
    ecg, _ = make_ecg(1_000.0, 6.0)

    detection = detect_r_peaks(ecg, 1_000.0, config=detector_config)

    minimum_distance = round(
        detector_config.candidate_refractory_seconds * 1_000.0
    )
    assert np.all(np.diff(detection.peak_samples) >= minimum_distance)


def test_event_selector_preserves_subphysiological_intervals_for_quality(
    detector_config: DetectorConfig,
) -> None:
    sampling_rate_hz = 1_000.0
    config = replace(detector_config, minimum_rr_seconds=0.6)
    samples = np.arange(3_000, dtype=float)
    signal = np.zeros(samples.size, dtype=float)
    for peak in (1_000, 1_500, 2_000):
        signal += np.exp(-0.5 * ((samples - peak) / 10.0) ** 2)
    template = np.exp(
        -0.5 * ((np.arange(600, dtype=float) - 200.0) / 10.0) ** 2
    )

    selection = _select_events(
        np.array([1_000, 1_500, 2_000]),
        signal,
        template,
        period=1.0,
        sampling_rate=sampling_rate_hz,
        config=config,
        polarity=1,
    )

    np.testing.assert_array_equal(selection.peaks, [1_000, 1_500, 2_000])
    assert selection.rejected_interval == 2

    quality = _quality_summary(
        candidate_count=3,
        selected_polarity=1,
        positive_candidate_count=3,
        negative_candidate_count=0,
        peak_samples=selection.peaks,
        correlations=selection.correlations,
        sampling_rate=sampling_rate_hz,
        rejected_low_prominence=0,
        rejected_low_correlation=0,
        rejected_double_mark=0,
        rejected_interval=selection.rejected_interval,
        config=config,
    )

    assert quality.degradation_reasons == ("rr_below_minimum",)
    assert quality.status == "degraded"


def test_quality_summary_reports_rr_reasons(
    detector_config: DetectorConfig,
) -> None:
    quality = _quality_summary(
        candidate_count=7,
        selected_polarity=1,
        positive_candidate_count=4,
        negative_candidate_count=3,
        peak_samples=np.array([0, 300, 1_200, 3_000]),
        correlations=np.ones(4),
        sampling_rate=1_000.0,
        rejected_low_prominence=0,
        rejected_low_correlation=0,
        rejected_double_mark=0,
        rejected_interval=0,
        config=detector_config,
    )

    assert quality.positive_candidate_count == 4
    assert quality.negative_candidate_count == 3
    assert quality.degradation_reasons == (
        "rr_below_minimum",
        "rr_above_maximum",
    )
    assert quality.status == "degraded"


def test_quality_summary_accepts_rr_boundaries(
    detector_config: DetectorConfig,
) -> None:
    quality = _quality_summary(
        candidate_count=3,
        selected_polarity=1,
        positive_candidate_count=3,
        negative_candidate_count=0,
        peak_samples=np.array([0, 400, 1_900]),
        correlations=np.ones(3),
        sampling_rate=1_000.0,
        rejected_low_prominence=0,
        rejected_low_correlation=0,
        rejected_double_mark=0,
        rejected_interval=0,
        config=detector_config,
    )

    assert quality.degradation_reasons == ()
    assert quality.status == "ok"


def test_detector_handles_amplitude_drift_and_deterministic_noise(
    detector_config: DetectorConfig,
) -> None:
    sampling_rate_hz = 1_000.0
    samples = np.arange(6_000, dtype=float)
    expected = np.rint(
        np.array([0.8, 1.65, 2.53, 3.44, 4.37, 5.31]) * sampling_rate_hz
    ).astype(np.int64)
    signal = np.zeros(samples.size, dtype=float)
    noise = np.random.default_rng(20260826).normal(0.0, 0.02, samples.size)
    for index, peak in enumerate(expected):
        amplitude = 0.75 + 0.1 * index
        signal += amplitude * np.exp(-0.5 * ((samples - peak) / 8.0) ** 2)
        signal += (
            0.65
            * amplitude
            * np.exp(-0.5 * ((samples - peak - 280.0) / 35.0) ** 2)
        )
    signal += 0.03 * np.sin(2.0 * np.pi * samples / (sampling_rate_hz * 7.0))
    signal += noise

    detection = detect_r_peaks(signal, sampling_rate_hz, config=detector_config)

    assert_peaks_match(detection.peak_samples, expected, tolerance_samples=10)


@pytest.mark.parametrize(
    "ecg",
    [
        np.zeros((2, 1000)),
        np.array([0.0, np.nan, 1.0]),
        np.array([True, False]),
    ],
)
def test_detector_rejects_invalid_ecg(
    detector_config: DetectorConfig,
    ecg: np.ndarray,
) -> None:
    with pytest.raises(CardiacInputError):
        detect_r_peaks(ecg, 1_000.0, config=detector_config)


def test_detector_rejects_invalid_sampling_rate(
    detector_config: DetectorConfig,
) -> None:
    ecg, _ = make_ecg(1_000.0, 6.0)

    with pytest.raises(CardiacInputError, match="sampling rate"):
        detect_r_peaks(ecg, 0.0, config=detector_config)
