import numpy as np
import pytest

from bcg_correction.bcg import (
    BcgCorrectionConfig,
    BcgInputError,
    _correct_aas,
    correct_bcg,
    rr_gap_spans,
)


def make_recording() -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    sampling_rate_hz = 1_000.0
    sample_count = 5_000
    samples = np.arange(sample_count, dtype=float)
    peak_samples = np.array([800, 1_600, 2_400, 3_200, 4_000], dtype=np.int64)
    artifact_anchors = peak_samples + 200
    clean = np.vstack(
        (
            2e-6 + 2e-7 * np.sin(2.0 * np.pi * samples / 173.0),
            -3e-6 + 3e-7 * np.cos(2.0 * np.pi * samples / 211.0),
            np.zeros(sample_count),
        )
    )
    artifact = np.exp(-0.5 * ((samples - 15.0) / 18.0) ** 2)
    data = clean.copy()
    for anchor in artifact_anchors:
        data[0] += 25e-6 * np.roll(artifact, anchor)
        data[1] += 12e-6 * np.roll(artifact, anchor)
    return data, clean, sampling_rate_hz, peak_samples


def correction_config(method: str) -> BcgCorrectionConfig:
    return BcgCorrectionConfig(
        method=method,
        window_seconds=(-0.1, 0.2),
        ecg_to_bcg_delay_seconds=0.2,
        aas_neighbor_count=2,
        pca_obs_components=1,
    )


@pytest.mark.parametrize("method", ["aas", "pca_obs"])
def test_correction_reduces_heartbeat_locked_artifact_and_preserves_boundaries(
    method: str,
) -> None:
    data, clean, sampling_rate_hz, peak_samples = make_recording()

    result = correct_bcg(
        data,
        peak_samples,
        sampling_rate_hz,
        channel_names=["EEG 001", "EEG 002", "ECG"],
        eeg_picks=np.array([0, 1], dtype=np.int64),
        ecg_channel_index=2,
        config=correction_config(method),
    )

    corrected = result.data_volts
    samples = result.corrected_samples
    before_error = np.sqrt(np.mean((data[:2, samples] - clean[:2, samples]) ** 2))
    after_error = np.sqrt(
        np.mean((corrected[:2, samples] - clean[:2, samples]) ** 2)
    )
    assert after_error < before_error
    outside = np.ones(data.shape[1], dtype=bool)
    outside[samples] = False
    np.testing.assert_array_equal(corrected[:, outside], data[:, outside])
    np.testing.assert_array_equal(corrected[2], data[2])
    assert result.method == method


def test_pca_obs_restores_input_means_outside_splice() -> None:
    data, _, sampling_rate_hz, peak_samples = make_recording()
    result = correct_bcg(
        data,
        peak_samples,
        sampling_rate_hz,
        channel_names=["EEG 001", "EEG 002", "ECG"],
        eeg_picks=np.array([0, 1], dtype=np.int64),
        ecg_channel_index=2,
        config=correction_config("pca_obs"),
    )

    outside = np.ones(data.shape[1], dtype=bool)
    outside[result.corrected_samples] = False
    np.testing.assert_array_equal(result.data_volts[:2, outside], data[:2, outside])
    np.testing.assert_allclose(
        result.data_volts[:2, outside].mean(axis=1),
        data[:2, outside].mean(axis=1),
        rtol=0.0,
        atol=0.0,
    )


def test_pca_obs_does_not_leave_a_demean_step_at_window_edges() -> None:
    """MNE demeans the whole recording; the splice must not import that offset."""
    data, _, sampling_rate_hz, peak_samples = make_recording()
    data = data + 80e-6
    result = correct_bcg(
        data,
        peak_samples,
        sampling_rate_hz,
        channel_names=["EEG 001", "EEG 002", "ECG"],
        eeg_picks=np.array([0, 1], dtype=np.int64),
        ecg_channel_index=2,
        config=correction_config("pca_obs"),
    )

    window_starts = result.corrected_samples[
        np.diff(result.corrected_samples, prepend=-2) != 1
    ]
    window_starts = window_starts[window_starts > 0]
    np.testing.assert_allclose(
        result.data_volts[:2, window_starts]
        - result.data_volts[:2, window_starts - 1],
        data[:2, window_starts] - data[:2, window_starts - 1],
        atol=5e-6,
    )


def test_edge_beats_are_dropped_instead_of_failing_the_run() -> None:
    data, _, sampling_rate_hz, peak_samples = make_recording()
    peak_samples = np.concatenate(
        (np.array([5], dtype=np.int64), peak_samples)
    )
    result = correct_bcg(
        data,
        peak_samples,
        sampling_rate_hz,
        channel_names=["EEG 001", "EEG 002", "ECG"],
        eeg_picks=np.array([0, 1], dtype=np.int64),
        ecg_channel_index=2,
        config=correction_config("aas"),
    )
    assert result.corrected_samples.size > 0
    assert result.corrected_samples.min() >= 0


def test_aas_scales_the_template_to_the_current_beat() -> None:
    data, clean, sampling_rate_hz, peak_samples = make_recording()
    samples = np.arange(data.shape[1], dtype=float)
    artifact = np.exp(-0.5 * ((samples - 15.0) / 18.0) ** 2)
    extra_anchor = int(peak_samples[2]) + 200
    data[0] += 25e-6 * np.roll(artifact, extra_anchor)

    result = correct_bcg(
        data,
        peak_samples,
        sampling_rate_hz,
        channel_names=["EEG 001", "EEG 002", "ECG"],
        eeg_picks=np.array([0, 1], dtype=np.int64),
        ecg_channel_index=2,
        config=correction_config("aas"),
    )

    span = slice(extra_anchor - 40, extra_anchor + 40)
    leftover = float(np.max(np.abs(result.data_volts[0, span] - clean[0, span])))
    unscaled = float(np.max(np.abs(data[0, span] - clean[0, span])))
    assert unscaled > 40e-6
    assert leftover < 0.25 * unscaled


def test_aas_rejects_a_dissimilar_neighbour_from_the_template() -> None:
    data, clean, sampling_rate_hz, peak_samples = make_recording()
    polluted_anchor = int(peak_samples[1]) + 200
    data[0, polluted_anchor - 25 : polluted_anchor + 25] += 400e-6

    result = correct_bcg(
        data,
        peak_samples,
        sampling_rate_hz,
        channel_names=["EEG 001", "EEG 002", "ECG"],
        eeg_picks=np.array([0, 1], dtype=np.int64),
        ecg_channel_index=2,
        config=correction_config("aas"),
    )

    target_anchor = int(peak_samples[2]) + 200
    span = slice(target_anchor - 40, target_anchor + 40)
    leftover = float(np.max(np.abs(result.data_volts[0, span] - clean[0, span])))
    assert leftover < 8e-6


def test_aas_preserves_complete_fixed_windows_across_irregular_rr() -> None:
    data = np.zeros((2, 8_500), dtype=np.float64)
    peaks = np.array([1_000, 1_800, 2_600, 3_400, 4_200, 8_000])
    artifact = np.sin(np.linspace(0.0, 4.0 * np.pi, 300))
    for peak in peaks:
        data[0, peak + 100 : peak + 400] = artifact

    result = correct_bcg(
        data,
        peaks,
        1_000.0,
        channel_names=["EEG", "ECG"],
        eeg_picks=np.array([0], dtype=np.int64),
        ecg_channel_index=1,
        config=correction_config("aas"),
    )

    expected = np.concatenate(
        [np.arange(peak + 100, peak + 400) for peak in peaks]
    )
    np.testing.assert_array_equal(result.corrected_samples, expected)


def test_aas_does_not_subtract_channel_baseline() -> None:
    data, _, sampling_rate_hz, peak_samples = make_recording()
    data = data + 80e-6
    result = correct_bcg(
        data,
        peak_samples,
        sampling_rate_hz,
        channel_names=["EEG 001", "EEG 002", "ECG"],
        eeg_picks=np.array([0, 1], dtype=np.int64),
        ecg_channel_index=2,
        config=correction_config("aas"),
    )

    window_starts = result.corrected_samples[
        np.diff(result.corrected_samples, prepend=-2) != 1
    ]
    window_starts = window_starts[window_starts > 0]
    np.testing.assert_allclose(
        result.data_volts[:2, window_starts]
        - result.data_volts[:2, window_starts - 1],
        data[:2, window_starts] - data[:2, window_starts - 1],
        atol=5e-6,
    )


def test_aas_fails_when_a_window_has_no_compatible_neighbor() -> None:
    data = np.zeros((2, 30), dtype=np.float64)
    bounds = ((0, 10), (10, 15), (15, 20))
    anchors = np.array([5, 12, 17], dtype=np.int64)

    with pytest.raises(BcgInputError, match="compatible neighbor"):
        _correct_aas(
            data,
            np.array([0], dtype=np.int64),
            bounds,
            anchors,
            neighbor_count=1,
        )


def test_aas_uses_exact_top_neighbors_without_an_arbitrary_cutoff() -> None:
    data = np.zeros((2, 30), dtype=np.float64)
    data[0, [1, 12, 23]] = 1.0
    bounds = ((0, 10), (10, 20), (20, 30))
    anchors = np.array([5, 15, 25], dtype=np.int64)

    corrected = _correct_aas(
        data,
        np.array([0], dtype=np.int64),
        bounds,
        anchors,
        neighbor_count=1,
    )

    assert corrected.shape == data.shape


def test_aas_subtraction_is_zero_at_every_window_boundary() -> None:
    sampling_rate_hz = 1_000.0
    sample_count = 5_000
    peak_samples = np.array([800, 1_600, 2_400, 3_200, 4_000])
    artifact_anchors = peak_samples + 200
    artifact = np.linspace(-20e-6, 30e-6, 300)
    data = np.zeros((2, sample_count), dtype=np.float64)
    for anchor in artifact_anchors:
        data[0, anchor - 100 : anchor + 200] += artifact

    result = correct_bcg(
        data,
        peak_samples,
        sampling_rate_hz,
        channel_names=["EEG", "ECG"],
        eeg_picks=np.array([0], dtype=np.int64),
        ecg_channel_index=1,
        config=correction_config("aas"),
    )

    starts = result.corrected_samples[
        np.diff(result.corrected_samples, prepend=-2) != 1
    ]
    stops = result.corrected_samples[
        np.diff(result.corrected_samples, append=sample_count + 1) != 1
    ]
    change = result.data_volts - data
    np.testing.assert_allclose(change[0, starts], 0.0, atol=1e-15)
    np.testing.assert_allclose(change[0, stops], 0.0, atol=1e-15)


def test_aas_rejects_a_window_shorter_than_two_samples() -> None:
    data = np.zeros((2, 100), dtype=np.float64)

    with pytest.raises(BcgInputError, match="at least two samples"):
        correct_bcg(
            data,
            np.array([20, 40, 60]),
            1_000.0,
            channel_names=["EEG", "ECG"],
            eeg_picks=np.array([0], dtype=np.int64),
            ecg_channel_index=1,
            config=BcgCorrectionConfig(
                method="aas",
                window_seconds=(0.0, 0.001),
                ecg_to_bcg_delay_seconds=0.0,
                aas_neighbor_count=2,
                pca_obs_components=1,
            ),
        )


def test_pca_obs_requires_effective_component_support() -> None:
    data, _, sampling_rate_hz, _ = make_recording()
    with pytest.raises(BcgInputError, match="n_components"):
        correct_bcg(
            data,
            np.array([800, 1_600], dtype=np.int64),
            sampling_rate_hz,
            channel_names=["EEG 001", "EEG 002", "ECG"],
            eeg_picks=np.array([0, 1], dtype=np.int64),
            ecg_channel_index=2,
            config=BcgCorrectionConfig(
                method="pca_obs",
                window_seconds=(-0.1, 0.2),
                ecg_to_bcg_delay_seconds=0.2,
                aas_neighbor_count=2,
                pca_obs_components=2,
            ),
        )


def test_rr_gap_spans_are_empty_when_intervals_are_in_bounds() -> None:
    peaks = np.array([1000, 1900, 2800, 3700], dtype=np.int64)

    assert rr_gap_spans(peaks, 1_000.0, 1.5) == ()


def test_rr_gap_spans_cover_the_full_long_interval() -> None:
    peaks = np.array([1000, 1900, 5200, 6100], dtype=np.int64)

    assert rr_gap_spans(peaks, 1_000.0, 1.5) == ((1900, 5200),)


def test_rr_gap_spans_keep_successive_gaps_separate() -> None:
    peaks = np.array([1000, 4000, 7000], dtype=np.int64)

    assert rr_gap_spans(peaks, 1_000.0, 1.5) == ((1000, 4000), (4000, 7000))
