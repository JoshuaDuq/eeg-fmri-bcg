"""Method-independent checks, including deliberately misleading corrections."""

import numpy as np
import pytest

from bcg_correction import correction_report


def test_local_rms_does_not_cancel_opposite_block_errors():
    from bcg_correction.evaluation import local_locked_rms

    epochs = np.ones((2, 40, 20))
    epochs[:, 20:] *= -1
    assert np.sqrt(np.mean(epochs.mean(axis=1) ** 2)) == 0
    np.testing.assert_allclose(local_locked_rms(epochs, 2), np.ones(2))


def test_local_rms_weights_unequal_blocks_by_beat_count():
    from bcg_correction.evaluation import local_locked_rms

    epochs = np.array([1, 1, 1, 3, 3], dtype=float)[None, :, None]
    np.testing.assert_allclose(local_locked_rms(epochs, 2), np.sqrt(21 / 5))


def test_local_rms_is_invariant_to_channel_polarity_and_order():
    from bcg_correction.evaluation import local_locked_rms

    epochs = np.random.default_rng(12).normal(size=(3, 40, 100))
    expected = local_locked_rms(epochs, 5)
    np.testing.assert_allclose(local_locked_rms(-epochs[::-1], 5), expected[::-1])


def test_epoch_spectrum_does_not_join_beat_boundaries():
    from scipy.signal import welch

    from bcg_correction.evaluation import epoch_spectrum

    epochs = np.random.default_rng(4).normal(size=(2, 12, 100))
    epochs += np.arange(12)[None, :, None] * 10
    frequency, expected = welch(epochs, fs=100, nperseg=100, axis=-1)
    actual_frequency, actual = epoch_spectrum(epochs, 100)
    np.testing.assert_allclose(actual_frequency, frequency)
    np.testing.assert_allclose(actual, expected.mean(axis=1))


def make_profile(method="aas", scale=0.4, subject="s1", label="run1"):
    from bcg_correction.evaluation import EvaluationSettings

    rng = np.random.default_rng(22)
    before = rng.normal(size=(4, 22000)) * 1e-6
    peaks = np.arange(100, 21900, 100)
    wave = np.sin(np.arange(60) / 9) * 20e-6
    for peak in peaks:
        before[:3, peak : peak + 60] += wave
    after = before.copy()
    after[:3] *= scale
    return correction_report.compute_correction_profile(
        before,
        after,
        ("Cz", "Pz", "Oz", "ECG"),
        ecg_channel_index=3,
        peak_samples=peaks,
        sampling_rate_hz=100,
        delay_seconds=0,
        window_seconds=(-0.2, 0.7),
        gap_fraction=0,
        method=method,
        label=label,
        subject=subject,
        evaluation=EvaluationSettings((2, 5, 10, 20), 8),
    )


def test_profile_is_method_label_invariant_and_scale_exact():
    from bcgnet.compare.arms import CLEAN_ARMS

    profiles = [make_profile(arm.key) for arm in CLEAN_ARMS]
    for profile in profiles:
        np.testing.assert_allclose(profile.local_ratio, 0.4, atol=1e-12)
        np.testing.assert_allclose(profile.local_ratio, profiles[0].local_ratio)


def test_zero_output_has_no_preservation_verdict():
    profile = make_profile(scale=0)
    np.testing.assert_allclose(profile.local_after_uv, 0, atol=1e-12)
    assert profile.preservation_status == "not_measured"


def test_grouped_summary_weights_participants_not_recordings():
    from bcg_correction.correction_report import participant_values

    profiles = [make_profile(scale=0.2, subject="s1", label=str(i)) for i in range(3)]
    profiles.append(make_profile(scale=0.8, subject="s2"))
    values = participant_values(profiles, "local_ratio")
    assert values.shape == (2, 2, 4)
    np.testing.assert_allclose(values.mean(axis=0), 0.5)


def test_invalid_pair_geometry_is_rejected():
    from bcg_correction.evaluation import EvaluationSettings

    with pytest.raises(ValueError, match="shape"):
        correction_report.compute_correction_profile(
            np.ones((3, 400)),
            np.ones((3, 399)),
            ("Cz", "Oz", "ECG"),
            ecg_channel_index=2,
            peak_samples=np.arange(50, 350, 50),
            sampling_rate_hz=100,
            delay_seconds=0,
            window_seconds=(-0.2, 0.7),
            gap_fraction=0,
            method="aas",
            evaluation=EvaluationSettings((2, 5), 2),
        )


def test_csv_exports_the_exact_profile_values_without_method_verdicts(tmp_path):
    import mne

    from bcg_correction.evaluation import EvaluationSettings
    from bcgnet.compare.arms import CLEAN_ARMS
    from bcgnet.compare.pairs import RecordingSet
    from bcgnet.compare.plots import metric_columns, metrics_row

    settings = EvaluationSettings((2, 5, 10, 20), 8)
    recording = RecordingSet("s1", "s1", "run1", 1, "run1", tmp_path / "a.vhdr", {})
    info = mne.create_info(["Cz", "ECG"], 100, ["eeg", "ecg"])
    data = np.random.default_rng(1).normal(size=(2, 1000)) * 1e-6
    raw = mne.io.RawArray(data, info, verbose="ERROR")
    profiles = {arm.key: make_profile(arm.key) for arm in CLEAN_ARMS}
    traces = {"Raw": raw, **{arm.label: raw for arm in CLEAN_ARMS}}
    row = metrics_row(recording, traces, profiles, max_hz=40, evaluation=settings)
    assert tuple(row) == metric_columns(settings)
    assert "prefer_comparator" not in row
    for arm in CLEAN_ARMS:
        assert (
            row[f"local_5_as_written_ratio_{arm.key}"]
            == profiles[arm.key].local_ratio[0, 1]
        )
        assert row[f"preservation_status_{arm.key}"] == "not_measured"
    expected_rms = np.sqrt(np.mean(data[0] ** 2)) * 1e6
    assert row["rms_raw"] == pytest.approx(expected_rms)


def test_pairing_uses_subject_and_unique_recording_before_aggregation():
    from bcgnet.compare.pipeline import paired_profiles

    first = make_profile(label="baseline")
    second = make_profile(label="run1")
    paired = paired_profiles({"aas": [first, second], "bcgnet": [second]})
    assert [p.label for p in paired["aas"]] == ["run1"]
    assert paired_profiles({"aas": [first], "bcgnet": []}) == {"aas": [], "bcgnet": []}


def test_missing_resolution_is_not_silently_dropped_from_cohort():
    from dataclasses import replace

    from bcg_correction.correction_report import participant_values

    profile = make_profile()
    missing = profile.local_ratio.copy()
    missing[:, -1] = np.nan
    second = replace(profile, label="run2", local_ratio=missing)
    values = participant_values([profile, second], "local_ratio")
    assert np.isnan(values[..., -1]).all()


def test_profile_scalar_export_has_no_nonfinite_json_numbers():
    import json

    from bcg_correction.correction_report import profile_metrics

    profile = make_profile(scale=1)
    values = profile_metrics(profile)
    assert values["locked_removal_fraction"] is None
    json.dumps(values, allow_nan=False)


def test_profiles_record_the_measurement_window_and_reject_mixing_it(tmp_path):
    from dataclasses import replace

    profile = make_profile()
    np.testing.assert_array_equal(profile.window_seconds, [-0.2, 0.7])
    other = replace(profile, label="run2", window_seconds=np.array([-0.1, 0.5]))
    with pytest.raises(ValueError, match="incompatible"):
        correction_report.save_aggregate_report(
            [profile, other], title="mixed", output=tmp_path / "mixed.png"
        )


def test_sparse_beats_are_explicitly_unavailable():
    from bcg_correction.evaluation import EvaluationSettings

    data = np.random.default_rng(9).normal(size=(2, 2100)) * 1e-6
    profile = correction_report.compute_correction_profile(
        data,
        data * 0.5,
        ("Cz", "ECG"),
        ecg_channel_index=1,
        peak_samples=np.arange(100, 2000, 100),
        sampling_rate_hz=100,
        delay_seconds=0,
        window_seconds=(-0.2, 0.7),
        gap_fraction=0,
        method="aas",
        evaluation=EvaluationSettings((2, 5), 8),
    )
    np.testing.assert_allclose(profile.local_ratio[:, 0], 0.5)
    assert np.isnan(profile.local_ratio[:, 1]).all()
    assert profile.block_minimum_beats.tolist() == [9, 3]


def test_regrid_to_anchor_places_the_bcg_peak_at_zero():
    """Stored waves are R-peak time; cohort plots must not smear different delays."""
    from bcg_correction.correction_report import (
        DISPLAY_MS,
        anchor_grid_ms,
        regrid_to_anchor,
    )

    delay = 0.2
    y = np.exp(-0.5 * ((DISPLAY_MS - delay * 1000.0) / 20.0) ** 2)
    grid = anchor_grid_ms((-0.2, 0.7))
    aligned = regrid_to_anchor(y, delay, grid)
    assert grid[int(np.nanargmax(aligned))] == pytest.approx(0.0, abs=1.0)


def test_regrid_to_anchor_does_not_invent_samples_outside_the_measured_span():
    from bcg_correction.correction_report import (
        DISPLAY_MS,
        anchor_grid_ms,
        regrid_to_anchor,
    )

    delay = 0.2
    y = np.full_like(DISPLAY_MS, np.nan)
    inside = (DISPLAY_MS >= 150.0) & (DISPLAY_MS <= 250.0)
    y[inside] = 1.0
    grid = anchor_grid_ms((-0.2, 0.7))
    aligned = regrid_to_anchor(y, delay, grid)
    assert np.isnan(aligned[grid < -50.5]).all()
    assert np.isnan(aligned[grid > 50.5]).all()
    assert np.isfinite(aligned[(grid >= -50.0) & (grid <= 50.0)]).all()


@pytest.mark.parametrize("scale", [0, 1])
def test_zero_and_identity_outputs_can_be_plotted_without_quality_verdicts(
    tmp_path, scale
):
    profile = make_profile(scale=scale)
    path = tmp_path / f"control_{scale}.png"
    correction_report.save_correction_report(profile, title="control", output=path)
    pages = correction_report.report_page_paths(path)
    assert pages.keys() == {"residual", "spectra", "ratios"}
    for page in pages.values():
        assert page.is_file()
        assert page.stat().st_size > 10_000


def test_phase_fraction_is_computed_before_participant_aggregation(
    tmp_path, monkeypatch
):
    from dataclasses import replace

    from bcg_correction.correction_report import participant_values

    profile = make_profile()
    first = replace(
        profile,
        phase_locking_spectrum=np.full_like(profile.phase_locking_spectrum, 0.1),
    )
    second = replace(
        profile,
        subject="s2",
        phase_locking_spectrum=np.ones_like(profile.phase_locking_spectrum),
    )
    first = replace(
        first,
        psd_removed_locked=np.ones_like(first.psd_removed_locked),
        psd_removed_variable=np.full_like(first.psd_removed_variable, 9),
    )
    second = replace(
        second,
        psd_removed_locked=np.full_like(second.psd_removed_locked, 100),
        psd_removed_variable=np.zeros_like(second.psd_removed_variable),
    )
    values = participant_values([first, second], "phase_locking_spectrum")
    np.testing.assert_allclose(np.median(values, axis=0), 0.55)
    plotted = []

    def capture(figure, *args, **kwargs):
        for axis in figure.axes:
            if "Locked" in (axis.get_ylabel() or ""):
                plotted.append(axis.lines[0].get_ydata())

    monkeypatch.setattr(correction_report, "save_figure", capture)
    correction_report.save_aggregate_report(
        [first, second], title="equal participants", output=tmp_path / "phase.png"
    )
    np.testing.assert_allclose(plotted[0], 0.55)


def test_frequency_panels_carry_one_hertz_minor_ticks(tmp_path, monkeypatch):
    """BCG energy is harmonic; the frequency axes must show a 1 Hz scale."""
    captured = []

    def capture(figure, *args, **kwargs):
        captured.append(figure)

    monkeypatch.setattr(correction_report, "save_figure", capture)
    first = make_profile(subject="s1")
    second = make_profile(subject="s2", label="r2")
    correction_report.save_aggregate_report(
        [first, second], title="ticks", output=tmp_path / "ticks.png"
    )
    def _axis(ylabel_part):
        for figure in captured:
            for axis in figure.axes:
                if ylabel_part in (axis.get_ylabel() or ""):
                    return axis
        raise AssertionError(ylabel_part)

    spectrum, phase = _axis("PSD"), _axis("Locked")
    from matplotlib.collections import LineCollection

    for axis in (spectrum, phase):
        majors = np.asarray(axis.get_xticks(minor=False), dtype=float)
        minors = np.asarray(axis.get_xticks(minor=True), dtype=float)
        ticks = np.unique(np.concatenate([majors, minors]))
        ticks = ticks[(ticks >= 1.0) & (ticks <= 10.0)]
        np.testing.assert_allclose(ticks, np.arange(1.0, 11.0))
        assert 5.0 in majors
        assert 1.0 in minors
        assert any(isinstance(artist, LineCollection) for artist in axis.collections)
    variable = np.median(
        np.stack([first.psd_removed_variable, second.psd_removed_variable]), axis=0
    )
    drawn = np.asarray(spectrum.lines[0].get_ydata(), dtype=float)
    expected = np.where(variable > 0, variable, np.nan)
    finite = np.isfinite(drawn) & np.isfinite(expected)
    np.testing.assert_allclose(drawn[finite], expected[finite])


def test_three_participants_draw_an_iqr_band(tmp_path, monkeypatch):
    profiles = [
        make_profile(subject="s1", label="a"),
        make_profile(subject="s2", label="b"),
        make_profile(subject="s3", label="c"),
    ]
    captured = []

    def capture(figure, *args, **kwargs):
        captured.append(figure)

    monkeypatch.setattr(correction_report, "save_figure", capture)
    correction_report.save_aggregate_report(
        profiles, title="iqr", output=tmp_path / "iqr.png"
    )
    from matplotlib.collections import PolyCollection

    def _has_iqr(ylabel_part=None, xlabel_part=None):
        for figure in captured:
            for axis in figure.axes:
                ylabel = axis.get_ylabel() or ""
                xlabel = axis.get_xlabel() or ""
                if ylabel_part and ylabel_part not in ylabel:
                    continue
                if xlabel_part and xlabel_part not in xlabel:
                    continue
                if any(
                    isinstance(artist, PolyCollection) for artist in axis.collections
                ):
                    return True
        return False

    assert _has_iqr(xlabel_part="BCG anchor")
    assert _has_iqr(ylabel_part="After / before")


def test_cohort_residual_aligns_heterogeneous_delays_to_the_bcg_anchor(
    tmp_path, monkeypatch
):
    from dataclasses import replace

    from bcg_correction.correction_report import DISPLAY_MS

    base = make_profile()

    def peaked(delay, subject):
        wave = np.exp(-0.5 * ((DISPLAY_MS - delay * 1000.0) / 25.0) ** 2) * 12.0
        return replace(
            base,
            subject=subject,
            label=subject,
            applied_delay_seconds=delay,
            local_wave_before=np.tile(wave, (len(base.block_counts), 1)),
            pooled_before=wave,
            local_wave_after=np.tile(wave * 0.4, (len(base.block_counts), 1)),
            pooled_after=wave * 0.4,
        )

    captured = []

    def capture(figure, *args, **kwargs):
        captured.append(figure)

    monkeypatch.setattr(correction_report, "save_figure", capture)
    correction_report.save_aggregate_report(
        [
            peaked(0.0, "s1"),
            peaked(0.2, "s2"),
            peaked(0.2, "s3"),
        ],
        title="aligned",
        output=tmp_path / "aligned.png",
    )
    residual = next(
        figure
        for figure in captured
        if any("BCG anchor" in (axis.get_xlabel() or "") for axis in figure.axes)
    )
    axis = residual.axes[0]
    x = np.asarray(axis.lines[0].get_xdata(), dtype=float)
    y = np.asarray(axis.lines[0].get_ydata(), dtype=float)
    finite = np.isfinite(y)
    peak_ms = float(x[finite][np.argmax(y[finite])])
    assert peak_ms == pytest.approx(0.0, abs=15.0)
    assert "BCG anchor" in (axis.get_xlabel() or "")


def test_eight_to_thirteen_hz_is_not_labelled_alpha(tmp_path, monkeypatch):
    captured = []

    def capture(figure, *args, **kwargs):
        captured.append(figure)

    monkeypatch.setattr(correction_report, "save_figure", capture)
    correction_report.save_aggregate_report(
        [make_profile()], title="band", output=tmp_path / "band.png"
    )
    texts = []
    for figure in captured:
        for axis in figure.axes:
            texts.extend(artist.get_text() for artist in axis.texts)
            texts.append(axis.get_title())
            texts.append(axis.get_ylabel())
            texts.append(axis.get_xlabel())
    joined = " ".join(texts)
    assert "8-13" in joined
    assert "$\\alpha$" not in joined
    assert "Beat-variable alpha" not in joined


def test_bcgnet_in_sample_note_appears_only_when_bcgnet_is_drawn(tmp_path, monkeypatch):
    from dataclasses import replace

    captured = []

    def capture(figure, *args, **kwargs):
        captured.append(" ".join(artist.get_text() for artist in figure.texts))

    monkeypatch.setattr(correction_report, "save_figure", capture)
    aas = make_profile(method="aas", subject="s1")
    correction_report.save_profile_report(
        {"AAS": [aas]}, title="aas only", output=tmp_path / "aas.png"
    )
    aas_text = " ".join(captured).lower()
    assert "train" not in aas_text

    captured.clear()
    bcgnet = replace(aas, method="bcgnet")
    correction_report.save_profile_report(
        {"AAS": [aas], "BCGNet": [bcgnet]},
        title="paired",
        output=tmp_path / "both.png",
    )
    both = " ".join(captured).lower()
    assert "train" in both
    assert "full recording" in both


def test_phase_spectrum_is_a_within_recording_energy_fraction():
    from bcg_correction.evaluation import spectral_locked_fraction

    locked = np.array([[1.0, 4.0], [3.0, 6.0]])
    variable = np.array([[4.0, 1.0], [2.0, 9.0]])
    np.testing.assert_allclose(spectral_locked_fraction(locked, variable), [0.4, 0.5])


def test_subject_coverage_counts_offered_not_largest_surviving_arm(
    tmp_path, monkeypatch
):
    from bcgnet.compare import pipeline

    a = make_profile(label="a")
    b = make_profile(label="b")
    c = make_profile(label="c")
    seen = []

    def capture(groups, **kwargs):
        seen.append(kwargs["coverage"])

    monkeypatch.setattr(pipeline, "save_comparative_report", capture)
    monkeypatch.setattr(
        correction_report, "save_topography_report", lambda *a, **k: False
    )
    pipeline._write_experiments(
        tmp_path, {"s1": {"aas": [a, b], "bcgnet": [b, c]}}, offered={"s1": 3}
    )
    assert seen[0] == {"aas": 1, "bcgnet": 1}


def test_no_paired_data_does_not_leave_old_pages_looking_current(tmp_path):
    from bcgnet.compare.pipeline import _write_experiments

    page = tmp_path / "cohort_comparative.png"
    page.write_bytes(b"old report")
    groups = {
        "s1": {"aas": [make_profile(label="a")], "bcgnet": [make_profile(label="b")]}
    }
    with pytest.raises(ValueError, match="stale"):
        _write_experiments(tmp_path, groups, offered={"s1": 2})
    assert page.read_bytes() == b"old report"
