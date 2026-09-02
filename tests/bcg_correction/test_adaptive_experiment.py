import json
import runpy
from dataclasses import astuple
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml


def test_experiment_manifest_serializes_numpy_scalars(tmp_path) -> None:
    script = Path(__file__).parents[2] / "tools" / "experiment_adaptive_bcg.py"
    write_json = runpy.run_path(str(script))["_write_json"]
    path = tmp_path / "run.json"

    write_json(path, {"sample_count": np.int64(593_101)})

    assert json.loads(path.read_text()) == {"sample_count": 593_101}


def test_experiment_config_resolves_paths_and_validates_the_sweep(tmp_path) -> None:
    script = Path(__file__).parents[2] / "tools" / "experiment_adaptive_bcg.py"
    load_config = runpy.run_path(str(script))["_load_experiment_config"]
    path = tmp_path / "experiment.yml"
    path.write_text(
        yaml.safe_dump(
            {
                "data_root": "data",
                "output_root": "results",
                "detector_config": "detector.yml",
                "recording_glob": "ThermalPainEEGFMRI_run3*_fastr.vhdr",
                "subjects": ["sub-0001", "sub-0002"],
                "sampling_rate_hz": 1000,
                "correction_window_seconds": [-0.2, 0.7],
                "cross_fit_fold_count": 10,
                "ecg_window_seconds": [-0.2, 0.4],
                "injection_amplitude_uv": 5.0,
                "injection_event_count": 120,
                "injection_event_width_seconds": 0.12,
                "injection_tone_frequencies_hz": [6.0, 10.0, 20.0],
                "neighbor_counts": [8, 20, 40],
                "morphology_component_counts": [4, 8, 16],
                "robust_morphology_components": 8,
                "morphology_samples": 96,
                "aas_neighbor_count": 20,
                "pca_obs_components": 2,
                "null_surrogate_count": 40,
                "random_seed": 20260831,
                "ridge_penalties": [0.1, 1.0, 10.0, 100.0],
            }
        )
    )

    config = load_config(path)

    assert config.data_root == tmp_path / "data"
    assert config.output_root == tmp_path / "results"
    assert config.recording_glob == "ThermalPainEEGFMRI_run3*_fastr.vhdr"
    assert config.subjects == ("sub-0001", "sub-0002")
    assert config.sampling_rate_hz == 1000.0
    assert config.neighbor_counts == (8, 20, 40)
    assert config.cross_fit_fold_count == 10
    assert config.injection_amplitude_uv == 5.0
    assert config.injection_event_count == 120
    assert config.injection_event_width_seconds == 0.12
    assert config.injection_tone_frequencies_hz == (6.0, 10.0, 20.0)
    assert config.ridge_penalties == (0.1, 1.0, 10.0, 100.0)
    assert config.aas_neighbor_count == 20
    assert config.pca_obs_components == 2


def test_experiment_config_rejects_unknown_study_knobs(tmp_path) -> None:
    script = Path(__file__).parents[2] / "tools" / "experiment_adaptive_bcg.py"
    load_config = runpy.run_path(str(script))["_load_experiment_config"]
    path = tmp_path / "experiment.yml"
    path.write_text(yaml.safe_dump({"surprise_threshold": 0.5}))

    with pytest.raises(ValueError, match="unknown keys"):
        load_config(path)


def test_candidate_sweep_keeps_neighbors_and_adds_all_ridge_models() -> None:
    script = Path(__file__).parents[2] / "tools" / "experiment_adaptive_bcg.py"
    candidates = runpy.run_path(str(script))["_candidates"]
    prepared = SimpleNamespace(
        temporal_features=np.arange(12, dtype=float).reshape(6, 2),
        rhythm_features=np.arange(18, dtype=float).reshape(6, 3),
        morphology_features=np.arange(96, dtype=float).reshape(6, 16),
    )
    experiment = SimpleNamespace(
        morphology_component_counts=(4, 8, 16),
        robust_morphology_components=8,
        neighbor_counts=(8, 20, 40),
        ridge_penalties=(0.1, 1.0, 10.0, 100.0),
    )

    sweep = candidates(prepared, experiment)

    assert len(sweep) == 39
    assert sum(candidate.aggregator == "weighted" for candidate in sweep) == 15
    assert sum(candidate.aggregator == "median" for candidate in sweep) == 3
    assert sum(candidate.aggregator == "ridge" for candidate in sweep) == 20
    assert sum(candidate.aggregator == "mean" for candidate in sweep) == 1
    assert {
        candidate.ridge_penalty
        for candidate in sweep
        if candidate.aggregator == "ridge"
    } == {0.1, 1.0, 10.0, 100.0}


def test_participant_workers_use_every_available_core_without_empty_jobs() -> None:
    script = Path(__file__).parents[2] / "tools" / "experiment_adaptive_bcg.py"
    worker_count = runpy.run_path(str(script))["_participant_worker_count"]

    assert worker_count(participant_count=12, available_cpu_count=10) == 10
    assert worker_count(participant_count=6, available_cpu_count=10) == 6


def test_tone_injection_uses_an_exact_fourier_frequency() -> None:
    script = Path(__file__).parents[2] / "tools" / "experiment_adaptive_bcg.py"
    tone_injection = runpy.run_path(str(script))["_tone_injection"]

    signal, exact_frequency = tone_injection(
        channel_count=3,
        sample_count=1_000,
        sampling_rate_hz=250.0,
        nominal_frequency_hz=10.1,
        amplitude_uv=5.0,
        generator=np.random.default_rng(3),
    )

    assert signal.shape == (3, 1_000)
    assert exact_frequency == 10.0
    assert np.sqrt(np.mean(signal**2)) == pytest.approx(5.0 / np.sqrt(2.0))


def test_event_injection_is_reproducible_and_nonzero() -> None:
    script = Path(__file__).parents[2] / "tools" / "experiment_adaptive_bcg.py"
    event_injection = runpy.run_path(str(script))["_event_injection"]
    arguments = {
        "channel_count": 3,
        "sample_count": 5_000,
        "sampling_rate_hz": 250.0,
        "event_count": 20,
        "width_seconds": 0.12,
        "amplitude_uv": 5.0,
    }

    first = event_injection(**arguments, generator=np.random.default_rng(4))
    repeated = event_injection(**arguments, generator=np.random.default_rng(4))

    np.testing.assert_array_equal(first, repeated)
    assert first.shape == (3, 5_000)
    assert np.all(np.isfinite(first))
    assert np.linalg.norm(first) > 0.0


def test_transfer_row_reports_known_tone_preservation() -> None:
    script = Path(__file__).parents[2] / "tools" / "experiment_adaptive_bcg.py"
    transfer_row = runpy.run_path(str(script))["_transfer_metric_row"]
    samples = np.arange(1_000) / 250.0
    signal = np.sin(2.0 * np.pi * 10.0 * samples)[np.newaxis, :]

    row = transfer_row(
        method="test",
        family="test",
        neighbor_count=None,
        aggregator="ridge",
        ridge_penalty=1.0,
        injection_name="tone_10hz",
        injected_uv=signal,
        corrected_increment_uv=signal,
        reference_ecg=np.cos(2.0 * np.pi * samples),
        sampling_rate_hz=250.0,
        nominal_frequency_hz=10.0,
        exact_frequency_hz=10.0,
    )

    for prefix in ("direct", "reference_orthogonalized"):
        assert row[f"{prefix}_gain"] == pytest.approx(1.0)
        assert row[f"{prefix}_relative_error"] == pytest.approx(0.0)
        assert row[f"{prefix}_cosine_similarity"] == pytest.approx(1.0)
        assert row[f"{prefix}_tone_amplitude_ratio_median"] == pytest.approx(1.0)
        assert row[
            f"{prefix}_tone_absolute_phase_error_degrees_median"
        ] == pytest.approx(0.0)
    assert "gain" not in row
    assert "relative_error" not in row


def test_metric_row_names_both_ecg_handling_variants_explicitly() -> None:
    from bcg_correction.adaptive import (
        ContinuousMetricVariants,
        EpochCorrectionMetrics,
    )

    script = Path(__file__).parents[2] / "tools" / "experiment_adaptive_bcg.py"
    metric_row = runpy.run_path(str(script))["_metric_variant_row"]
    direct = EpochCorrectionMetrics(
        locked_ratio=0.4,
        held_out_ratio=0.9,
        locked_before=10.0,
        locked_after=4.0,
        specificity=0.8,
        alpha_collateral_fraction=0.2,
        locked_removed=6.0,
        collateral=1.0,
    )
    projected = EpochCorrectionMetrics(
        locked_ratio=0.2,
        held_out_ratio=1.1,
        locked_before=8.0,
        locked_after=1.6,
        specificity=0.9,
        alpha_collateral_fraction=0.1,
        locked_removed=6.4,
        collateral=0.5,
    )

    row = metric_row(
        "test",
        "test",
        None,
        "mean",
        None,
        ContinuousMetricVariants(
            direct=direct,
            reference_orthogonalized=projected,
        ),
    )

    assert row["direct_locked_ratio"] == 0.4
    assert row["reference_orthogonalized_locked_ratio"] == 0.2
    assert row["direct_held_out_log_distortion"] == pytest.approx(abs(np.log(0.9)))
    assert row["reference_orthogonalized_held_out_log_distortion"] == pytest.approx(
        abs(np.log(1.1))
    )
    assert row["direct_diagnostic_specificity"] == 0.8
    assert row["direct_diagnostic_alpha_collateral_fraction"] == 0.2
    assert "direct_specificity" not in row
    assert "direct_alpha_collateral_fraction" not in row
    assert "locked_ratio" not in row
    assert "held_out_ratio" not in row


def test_cohort_summary_keeps_metric_variants_separate_without_selection() -> None:
    from bcg_correction.adaptive import (
        ContinuousMetricVariants,
        EpochCorrectionMetrics,
    )

    script = Path(__file__).parents[2] / "tools" / "experiment_adaptive_bcg.py"
    module = runpy.run_path(str(script))
    metric_row = module["_metric_variant_row"]
    summarize = module["_summarize_cohort"]

    def metrics(locked_ratio: float) -> EpochCorrectionMetrics:
        return EpochCorrectionMetrics(
            locked_ratio=locked_ratio,
            held_out_ratio=1.0,
            locked_before=10.0,
            locked_after=10.0 * locked_ratio,
            specificity=0.8,
            alpha_collateral_fraction=0.2,
            locked_removed=5.0,
            collateral=1.0,
        )

    rows = []
    for subject, direct, projected in (
        ("sub-1", 0.4, 0.2),
        ("sub-2", 0.6, 0.3),
    ):
        rows.append(
            {"subject": subject}
            | metric_row(
                "test",
                "test",
                None,
                "mean",
                None,
                ContinuousMetricVariants(
                    direct=metrics(direct),
                    reference_orthogonalized=metrics(projected),
                ),
            )
        )

    summary = summarize(rows, seed=7)[0]

    assert summary["direct_locked_ratio_median"] == pytest.approx(0.5)
    assert summary["reference_orthogonalized_locked_ratio_median"] == pytest.approx(
        0.25
    )
    assert "locked_ratio_median" not in summary
    assert not any("pareto" in key for key in summary)


def test_metric_audit_plot_compares_explicit_variants(tmp_path) -> None:
    script = Path(__file__).parents[2] / "tools" / "experiment_adaptive_bcg.py"
    plot = runpy.run_path(str(script))["_plot_metric_audit"]
    output = tmp_path / "audit.png"
    rows = [
        {
            "method": "blocked_mean",
            "family": "blocked_mean",
            "aggregator": "mean",
            "direct_locked_ratio_median": 0.3,
            "reference_orthogonalized_locked_ratio_median": 0.2,
            "direct_held_out_log_distortion_median": 0.1,
            "reference_orthogonalized_held_out_log_distortion_median": 0.05,
        }
    ]

    plot(rows, output)

    assert output.is_file()
    assert output.stat().st_size > 0


def test_experiment_scores_from_original_eeg_before_ecg_projection() -> None:
    from bcg_correction.adaptive import continuous_epoch_metric_variants

    script = Path(__file__).parents[2] / "tools" / "experiment_adaptive_bcg.py"
    score = runpy.run_path(str(script))["_score_correction_variants"]
    sampling_rate = 100.0
    epoch_samples = 100
    starts = np.arange(8, dtype=np.int64) * epoch_samples
    samples = np.arange(8 * epoch_samples) / sampling_rate
    ecg = np.sin(2.0 * np.pi * samples)
    locked = np.tile(np.sin(4.0 * np.pi * samples[:epoch_samples]), 8)
    modulation = np.repeat(np.linspace(-1.0, 1.0, 8), epoch_samples)
    raw_eeg = np.vstack(
        (locked + 3.0 * ecg + modulation, 0.5 * locked - ecg - modulation)
    )
    selected_volts = np.vstack((raw_eeg, ecg)) * 1e-6
    corrected_uv = np.vstack(
        (
            0.4 * locked + ecg + modulation,
            0.2 * locked - 0.2 * ecg - modulation,
        )
    )

    actual = score(
        selected_volts,
        corrected_uv,
        starts,
        epoch_samples=epoch_samples,
        sampling_rate_hz=sampling_rate,
    )
    expected = continuous_epoch_metric_variants(
        raw_eeg,
        corrected_uv,
        ecg,
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


def test_rank_one_candidate_transfer_matches_full_multichannel_correction() -> None:
    from bcg_correction.adaptive import (
        apply_template_predictions_to_recording,
        predict_cross_fitted_templates,
    )

    script = Path(__file__).parents[2] / "tools" / "experiment_adaptive_bcg.py"
    module = runpy.run_path(str(script))
    correct_rank_one = module["_correct_candidate_injection"]
    Candidate = module["Candidate"]
    Injection = module["Injection"]
    starts = np.array([0, 10, 20, 30], dtype=np.int64)
    epoch_samples = 6
    temporal = np.sin(np.arange(40) / 3.0)
    topography = np.array([1.0, -0.5, 2.0])
    signal = topography[:, np.newaxis] * temporal
    features = np.arange(4, dtype=float)[:, np.newaxis]
    training_mask = ~np.eye(4, dtype=bool)
    candidate = Candidate(
        name="test",
        family="test",
        aggregator="weighted",
        features=features,
        neighbor_count=2,
        ridge_penalty=None,
    )
    injection = Injection(
        name="test",
        signal_uv=signal,
        topography=topography,
        temporal_uv=temporal,
        nominal_frequency_hz=None,
        exact_frequency_hz=None,
    )
    indices = starts[:, np.newaxis] + np.arange(epoch_samples)
    epochs = signal[:, indices].transpose(1, 0, 2)
    templates = epochs - epochs.mean(axis=2, keepdims=True)
    predictions = predict_cross_fitted_templates(
        templates,
        features,
        training_mask,
        neighbor_count=2,
    )
    expected = apply_template_predictions_to_recording(
        signal,
        starts,
        predictions,
    )

    actual = correct_rank_one(
        candidate,
        injection,
        prepared_starts=starts,
        epoch_samples=epoch_samples,
        training_mask=training_mask,
    )

    np.testing.assert_allclose(actual, expected, atol=1e-12)
