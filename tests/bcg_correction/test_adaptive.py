import warnings
from dataclasses import astuple

import numpy as np

from bcg_correction.adaptive import (
    AdaptiveEpochConfig,
    apply_template_predictions_to_recording,
    combine_feature_groups,
    contiguous_cross_fit_training_mask,
    continuous_epoch_metric_variants,
    epoch_correction_metrics,
    predict_cross_fitted_mean_templates,
    predict_cross_fitted_median_templates,
    predict_cross_fitted_reference_residual_mean_templates,
    predict_cross_fitted_ridge_templates,
    predict_cross_fitted_templates,
    prepare_beat_epochs,
)


def test_prediction_for_a_beat_never_uses_that_beats_eeg() -> None:
    epochs = np.array(
        [
            [[1.0, 2.0, 1.0, 0.0]],
            [[1.1, 2.1, 1.1, 0.1]],
            [[0.9, 1.9, 0.9, -0.1]],
            [[1.2, 2.2, 1.2, 0.2]],
        ]
    )
    features = np.arange(4, dtype=np.float64)[:, np.newaxis]
    training_mask = ~np.eye(4, dtype=bool)
    original = predict_cross_fitted_templates(
        epochs,
        features,
        training_mask,
        neighbor_count=2,
    )

    changed = epochs.copy()
    changed[1] += 10_000.0
    repeated = predict_cross_fitted_templates(
        changed,
        features,
        training_mask,
        neighbor_count=2,
    )

    np.testing.assert_array_equal(repeated[1], original[1])


def test_ridge_prediction_for_a_beat_never_uses_that_beats_eeg() -> None:
    features = np.linspace(-2.0, 2.0, 8)[:, np.newaxis]
    shape = np.array([0.0, 1.0, -1.0, 0.0])
    epochs = (3.0 + 2.0 * features[:, 0])[:, np.newaxis, np.newaxis] * shape
    training_mask = ~np.eye(8, dtype=bool)
    original = predict_cross_fitted_ridge_templates(
        epochs,
        features,
        training_mask,
        ridge_penalty=1e-6,
    )

    changed = epochs.copy()
    changed[3] += 10_000.0
    repeated = predict_cross_fitted_ridge_templates(
        changed,
        features,
        training_mask,
        ridge_penalty=1e-6,
    )

    np.testing.assert_allclose(repeated[3], original[3], atol=1e-8)
    np.testing.assert_allclose(original, epochs, atol=1e-5)


def test_ridge_large_output_does_not_emit_matmul_runtime_warnings() -> None:
    generator = np.random.default_rng(7)
    epochs = generator.normal(size=(40, 1, 200))
    features = np.linspace(-1.0, 1.0, 40)[:, np.newaxis]
    training_mask = ~np.eye(40, dtype=bool)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        predicted = predict_cross_fitted_ridge_templates(
            epochs,
            features,
            training_mask,
            ridge_penalty=1.0,
        )

    assert np.all(np.isfinite(predicted))


def test_block_mean_uses_only_eligible_training_epochs() -> None:
    epochs = np.arange(6 * 4, dtype=float).reshape(6, 1, 4)
    training_mask = np.array(
        [
            [False, False, True, True, True, True],
            [False, False, True, True, True, True],
            [True, True, False, False, True, True],
            [True, True, False, False, True, True],
            [True, True, True, True, False, False],
            [True, True, True, True, False, False],
        ]
    )

    predicted = predict_cross_fitted_mean_templates(epochs, training_mask)

    np.testing.assert_array_equal(predicted[0], epochs[2:].mean(axis=0))
    changed = epochs.copy()
    changed[:2] += 10_000.0
    repeated = predict_cross_fitted_mean_templates(changed, training_mask)
    np.testing.assert_array_equal(repeated[:2], predicted[:2])


def test_conditioning_features_select_matching_artifact_morphology() -> None:
    first_shape = np.array([0.0, 1.0, 0.0, -1.0])
    second_shape = np.array([0.0, -1.0, 0.0, 1.0])
    epochs = np.stack(
        [first_shape, second_shape, first_shape, second_shape, first_shape],
        axis=0,
    )[:, np.newaxis, :]
    morphology = np.array([[0.0], [1.0], [0.0], [1.0], [0.0]])

    predicted = predict_cross_fitted_templates(
        epochs,
        morphology,
        ~np.eye(5, dtype=bool),
        neighbor_count=1,
    )

    np.testing.assert_allclose(predicted[:, 0], epochs[:, 0], atol=1e-12)


def test_prediction_rejects_too_few_independent_neighbours() -> None:
    epochs = np.zeros((3, 2, 10), dtype=np.float64)
    features = np.zeros((3, 1), dtype=np.float64)

    with np.testing.assert_raises_regex(
        ValueError,
        "neighbor_count must be smaller than the beat count",
    ):
        predict_cross_fitted_templates(
            epochs,
            features,
            ~np.eye(3, dtype=bool),
            neighbor_count=3,
        )


def test_prepared_features_separate_ecg_morphologies() -> None:
    sampling_rate_hz = 100.0
    peak_samples = np.arange(100, 900, 100, dtype=np.int64)
    sample_count = 1_000
    eeg = np.zeros((2, sample_count), dtype=np.float64)
    ecg = np.zeros(sample_count, dtype=np.float64)
    shape = np.sin(np.linspace(0.0, 2.0 * np.pi, 30, endpoint=False))
    for beat_index, peak in enumerate(peak_samples):
        sign = 1.0 if beat_index % 2 == 0 else -1.0
        ecg[peak - 10 : peak + 20] = sign * shape
        eeg[:, peak : peak + 30] = sign

    prepared = prepare_beat_epochs(
        eeg,
        ecg,
        peak_samples,
        sampling_rate_hz,
        config=AdaptiveEpochConfig(
            correction_window_seconds=(-0.1, 0.2),
            ecg_window_seconds=(-0.1, 0.2),
            ecg_to_bcg_delay_seconds=0.1,
            morphology_components=2,
            morphology_samples=20,
        ),
    )

    assert prepared.eeg_epochs.shape == (8, 2, 30)
    assert prepared.rhythm_features.shape == (8, 3)
    assert prepared.temporal_features.shape == (8, 1)
    same = np.linalg.norm(
        prepared.morphology_features[0] - prepared.morphology_features[2]
    )
    different = np.linalg.norm(
        prepared.morphology_features[0] - prepared.morphology_features[1]
    )
    assert same < different


def test_feature_groups_contribute_equal_average_squared_scale() -> None:
    one_dimension = np.arange(6, dtype=np.float64)[:, np.newaxis]
    repeated = np.repeat(one_dimension, 9, axis=1)

    combined = combine_feature_groups(one_dimension, repeated)

    first_group_variance = np.mean(combined[:, :1] ** 2)
    second_group_variance = np.sum(combined[:, 1:] ** 2, axis=1).mean()
    np.testing.assert_allclose(first_group_variance, second_group_variance)


def test_continuous_correction_averages_overlapping_predictions() -> None:
    recording = np.zeros((1, 8), dtype=np.float64)
    starts = np.array([1, 2], dtype=np.int64)
    predictions = np.array(
        [
            [[2.0, 2.0, 2.0, 2.0]],
            [[4.0, 4.0, 4.0, 4.0]],
        ]
    )

    corrected = apply_template_predictions_to_recording(
        recording,
        starts,
        predictions,
    )

    np.testing.assert_allclose(
        corrected[0],
        [0.0, 0.0, -1.0, -3.0, -2.0, 0.0, 0.0, 0.0],
    )


def test_metrics_expose_collateral_instead_of_rewarding_extra_subtraction() -> None:
    sampling_rate_hz = 100.0
    samples = np.arange(100, dtype=np.float64)
    artifact = 8.0 * np.sin(2.0 * np.pi * 2.0 * samples / sampling_rate_hz)
    alpha = np.stack(
        [
            2.0
            * np.sin(
                2.0 * np.pi * 10.0 * samples / sampling_rate_hz + phase
            )
            for phase in np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
        ]
    )[:, np.newaxis, :]
    before = artifact[np.newaxis, np.newaxis, :] + alpha
    artifact_only = alpha
    overcorrected = np.zeros_like(before)

    specific = epoch_correction_metrics(
        before,
        artifact_only,
        sampling_rate_hz=sampling_rate_hz,
    )
    aggressive = epoch_correction_metrics(
        before,
        overcorrected,
        sampling_rate_hz=sampling_rate_hz,
    )

    assert aggressive.locked_ratio <= specific.locked_ratio
    assert specific.specificity > aggressive.specificity
    assert specific.alpha_collateral_fraction < aggressive.alpha_collateral_fraction


def test_metrics_measure_held_out_beat_variation() -> None:
    generator = np.random.default_rng(9)
    before = generator.normal(size=(12, 2, 100))
    after = 0.4 * before

    metrics = epoch_correction_metrics(
        before,
        after,
        sampling_rate_hz=100.0,
    )

    np.testing.assert_allclose(metrics.held_out_ratio, 0.4, rtol=1e-12)


def test_metric_variants_transform_before_and_after_symmetrically() -> None:
    sampling_rate = 100.0
    epoch_samples = 100
    starts = np.arange(8, dtype=np.int64) * epoch_samples
    samples = np.arange(8 * epoch_samples, dtype=np.float64)
    reference = np.sin(2.0 * np.pi * samples / sampling_rate)
    locked = np.tile(
        np.sin(4.0 * np.pi * np.arange(epoch_samples) / sampling_rate),
        8,
    )
    modulation = np.repeat(np.linspace(-1.0, 1.0, 8), epoch_samples)
    before = np.vstack(
        (
            locked + 3.0 * reference + modulation,
            0.5 * locked - 2.0 * reference - modulation,
        )
    )
    after = np.vstack(
        (
            0.4 * locked + 1.0 * reference + modulation,
            0.2 * locked - 0.5 * reference - modulation,
        )
    )

    variants = continuous_epoch_metric_variants(
        before,
        after,
        reference,
        starts,
        epoch_samples=epoch_samples,
        sampling_rate_hz=sampling_rate,
    )

    centred_before = before - np.median(before, axis=1, keepdims=True)
    centred_after = after - np.median(after, axis=1, keepdims=True)
    direct_before = centred_before[
        :, starts[:, None] + np.arange(epoch_samples)
    ].transpose(1, 0, 2)
    direct_after = centred_after[
        :, starts[:, None] + np.arange(epoch_samples)
    ].transpose(1, 0, 2)
    expected_direct = epoch_correction_metrics(
        direct_before,
        direct_after,
        sampling_rate_hz=sampling_rate,
    )
    from bcg_correction.metrics import regress_out_reference

    projected_before = regress_out_reference(before, reference)
    projected_after = regress_out_reference(after, reference)
    projected_before -= np.median(projected_before, axis=1, keepdims=True)
    projected_after -= np.median(projected_after, axis=1, keepdims=True)
    projected_before_epochs = projected_before[
        :, starts[:, None] + np.arange(epoch_samples)
    ].transpose(1, 0, 2)
    projected_after_epochs = projected_after[
        :, starts[:, None] + np.arange(epoch_samples)
    ].transpose(1, 0, 2)
    expected_projected = epoch_correction_metrics(
        projected_before_epochs,
        projected_after_epochs,
        sampling_rate_hz=sampling_rate,
    )

    assert variants.direct == expected_direct
    assert variants.reference_orthogonalized == expected_projected


def test_metric_variants_ignore_recording_level_channel_offsets() -> None:
    sampling_rate = 100.0
    epoch_samples = 100
    starts = np.arange(8, dtype=np.int64) * epoch_samples
    samples = np.arange(8 * epoch_samples, dtype=np.float64)
    reference = np.sin(2.0 * np.pi * samples / sampling_rate)
    locked = np.tile(
        np.sin(4.0 * np.pi * np.arange(epoch_samples) / sampling_rate),
        8,
    )
    modulation = np.repeat(np.linspace(-1.0, 1.0, 8), epoch_samples)
    before = np.vstack((locked + modulation, 0.5 * locked - modulation))
    after = np.vstack((0.4 * locked + modulation, 0.2 * locked - modulation))

    expected = continuous_epoch_metric_variants(
        before,
        after,
        reference,
        starts,
        epoch_samples=epoch_samples,
        sampling_rate_hz=sampling_rate,
    )
    actual = continuous_epoch_metric_variants(
        before + np.array([[100.0], [-75.0]]),
        after + np.array([[-30.0], [250.0]]),
        reference,
        starts,
        epoch_samples=epoch_samples,
        sampling_rate_hz=sampling_rate,
    )

    np.testing.assert_allclose(
        astuple(actual.direct),
        astuple(expected.direct),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        astuple(actual.reference_orthogonalized),
        astuple(expected.reference_orthogonalized),
        atol=1e-12,
    )


def test_median_template_resists_a_transient_in_one_neighbour() -> None:
    epochs = np.zeros((5, 1, 7), dtype=np.float64)
    epochs[2, 0, 3] = 1_000.0
    features = np.zeros((5, 1), dtype=np.float64)

    predicted = predict_cross_fitted_median_templates(
        epochs,
        features,
        ~np.eye(5, dtype=bool),
        neighbor_count=3,
    )

    np.testing.assert_array_equal(predicted[0], 0.0)


def test_contiguous_cross_fit_excludes_blocks_and_physical_overlap() -> None:
    starts = np.array([0, 10, 20, 30, 40, 50], dtype=np.int64)

    mask = contiguous_cross_fit_training_mask(
        starts,
        epoch_samples=15,
        fold_count=3,
    )

    assert not mask[0, 1]
    assert not mask[1, 0]
    assert not mask[1, 2]
    assert mask[0, 2]
    assert mask[0, 4]


def test_block_predictions_ignore_all_eeg_in_the_held_out_block() -> None:
    epochs = np.arange(8 * 4, dtype=np.float64).reshape(8, 1, 4)
    features = np.arange(8, dtype=np.float64)[:, np.newaxis]
    mask = contiguous_cross_fit_training_mask(
        np.arange(8, dtype=np.int64) * 10,
        epoch_samples=4,
        fold_count=4,
    )
    original = predict_cross_fitted_templates(
        epochs,
        features,
        mask,
        neighbor_count=2,
    )

    changed = epochs.copy()
    changed[2:4] += 10_000.0
    repeated = predict_cross_fitted_templates(
        changed,
        features,
        mask,
        neighbor_count=2,
    )

    np.testing.assert_array_equal(repeated[2:4], original[2:4])


def test_reference_residual_mean_uses_training_beats_only() -> None:
    eeg_epochs = np.arange(8 * 2 * 5, dtype=np.float64).reshape(8, 2, 5)
    reference_epochs = np.sin(
        np.arange(8 * 5, dtype=np.float64).reshape(8, 5)
    )
    mask = contiguous_cross_fit_training_mask(
        np.arange(8, dtype=np.int64) * 10,
        epoch_samples=5,
        fold_count=4,
    )
    original = predict_cross_fitted_reference_residual_mean_templates(
        eeg_epochs,
        reference_epochs,
        mask,
    )

    changed = eeg_epochs.copy()
    changed[2:4] += 10_000.0
    repeated = predict_cross_fitted_reference_residual_mean_templates(
        changed,
        reference_epochs,
        mask,
    )

    np.testing.assert_array_equal(repeated[2:4], original[2:4])
