"""Paired simulator-vs-scanner thermal 8-13 Hz response, not absolute power."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def _protocol_values(**overrides) -> dict:
    values = {
        "runs": [1, 2, 3, 4, 5, 6],
        "n_triggers": 11,
        "prestimulus_seconds": [-5.0, 0.0],
        "ramp_up_seconds": 3.0,
        "hold_seconds": 7.5,
        "posterior": [
            "P3",
            "P4",
            "Pz",
            "PO3",
            "PO4",
            "PO7",
            "PO8",
            "POz",
            "O1",
            "O2",
            "Oz",
        ],
        "band_hz": [8.0, 13.0],
        "peak_to_peak_uv": 200.0,
        "welch_nperseg_seconds": 1.0,
    }
    values.update(overrides)
    return values


def _protocol(**overrides):
    from bcgstudy.thermal_response import ThermalProtocol

    return ThermalProtocol.from_mapping(_protocol_values(**overrides))


def test_thermal_protocol_comes_from_config() -> None:
    from bcgstudy.thermal_response import ThermalProtocol

    protocol = ThermalProtocol.from_mapping(
        {
            "runs": [1, 3],
            "n_triggers": 9,
            "prestimulus_seconds": [-4.0, 0.0],
            "ramp_up_seconds": 2.5,
            "hold_seconds": 6.0,
            "posterior": ["Pz", "Oz"],
            "band_hz": [9.0, 12.0],
            "peak_to_peak_uv": 175.0,
            "welch_nperseg_seconds": 2.0,
        }
    )

    assert protocol.runs == (1, 3)
    assert protocol.n_triggers == 9
    assert protocol.prestimulus_seconds == (-4.0, 0.0)
    assert protocol.plateau_seconds == (2.5, 8.5)
    assert protocol.posterior_channels == ("Pz", "Oz")
    assert protocol.band_hz == (9.0, 12.0)
    assert protocol.peak_to_peak_uv == 175.0
    assert protocol.welch_nperseg_seconds == 2.0


def test_thermal_protocol_rejects_missing_or_invalid_values() -> None:
    from bcgstudy.thermal_response import ThermalProtocol

    with pytest.raises(ValueError, match="missing thermal settings"):
        ThermalProtocol.from_mapping({"runs": [1]})

    values = {
        "runs": [1],
        "n_triggers": 11,
        "prestimulus_seconds": [-5.0, 0.0],
        "ramp_up_seconds": 3.0,
        "hold_seconds": 7.5,
        "posterior": ["Pz", "Oz"],
        "band_hz": [13.0, 8.0],
        "peak_to_peak_uv": 200.0,
        "welch_nperseg_seconds": 1.0,
    }
    with pytest.raises(ValueError, match="band_hz must be increasing"):
        ThermalProtocol.from_mapping(values)


def test_plateau_window_is_the_documented_hold_not_an_eeg_peek() -> None:
    protocol = _protocol()
    assert protocol.prestimulus_seconds == (-5.0, 0.0)
    assert protocol.plateau_seconds == (3.0, 10.5)


def test_response_ratio_is_ten_log10_plateau_over_prestim() -> None:
    from bcgstudy.thermal_response import response_ratio_db

    assert response_ratio_db(2.0, 1.0) == pytest.approx(10.0 * np.log10(2.0))
    assert response_ratio_db(1.0, 1.0) == pytest.approx(0.0)
    assert response_ratio_db(0.25, 1.0) == pytest.approx(-6.020599913279624)
    assert np.isnan(response_ratio_db(1.0, 0.0))
    assert np.isnan(response_ratio_db(-1.0, 1.0))


def test_halving_10hz_amplitude_gives_minus_six_db_via_compute_psd() -> None:
    from bcgstudy.thermal_response import epoch_band_power, response_ratio_db

    protocol = _protocol()
    sfreq = 1000.0
    prestim = _sine_epochs(amplitude=2.0, seconds=5.0, sfreq=sfreq)
    plateau = _sine_epochs(amplitude=1.0, seconds=7.5, sfreq=sfreq)
    picks = list(range(len(protocol.posterior_channels)))
    p_pre = epoch_band_power(
        prestim,
        sfreq=sfreq,
        band=protocol.band_hz,
        nperseg_seconds=protocol.welch_nperseg_seconds,
        picks=picks,
    )
    p_plat = epoch_band_power(
        plateau,
        sfreq=sfreq,
        band=protocol.band_hz,
        nperseg_seconds=protocol.welch_nperseg_seconds,
        picks=picks,
    )
    ratio = response_ratio_db(float(np.median(p_plat)), float(np.median(p_pre)))
    assert ratio == pytest.approx(-6.020599913279624, abs=0.4)


def test_median_absolute_error_is_over_trials_not_a_grand_mean() -> None:
    from bcgstudy.thermal_response import (
        median_absolute_error,
        signed_median_difference,
    )

    simulator = np.array([-1.0, -2.0, -3.0])
    method = np.array([-1.0, 0.0, -3.0])
    assert median_absolute_error(method, simulator) == pytest.approx(0.0)
    method = np.array([-1.5, -2.5, -3.5])
    assert median_absolute_error(method, simulator) == pytest.approx(0.5)
    assert signed_median_difference(method, simulator) == pytest.approx(-0.5)


def test_participant_errors_do_not_treat_trials_as_independent_subjects() -> None:
    from bcgstudy.thermal_response import participant_method_summaries

    rows = []
    for trial in range(11):
        rows.append(
            {
                "bids_id": "sub-0019",
                "run": 1,
                "trial": trial + 1,
                "method": "aas",
                "r_method": -1.0 if trial else 10.0,
                "r_simulator": -1.0,
            }
        )
    rows.append(
        {
            "bids_id": "sub-0020",
            "run": 1,
            "trial": 1,
            "method": "aas",
            "r_method": -4.0,
            "r_simulator": -1.0,
        }
    )
    summary = participant_method_summaries(rows)
    by_subject = {row["bids_id"]: row for row in summary if row["method"] == "aas"}
    assert set(by_subject) == {"sub-0019", "sub-0020"}
    assert by_subject["sub-0019"]["median_absolute_error"] == pytest.approx(0.0)
    assert by_subject["sub-0020"]["median_absolute_error"] == pytest.approx(3.0)
    assert by_subject["sub-0020"]["signed_median_difference"] == pytest.approx(-3.0)


def test_completed_thermal_keeps_eleven_triggers_and_rejects_aborted() -> None:
    from bcgstudy.thermal_response import choose_completed_thermal

    aborted = _candidate(n_triggers=0, duration_seconds=2.0, name="run1_abort")
    complete = _candidate(n_triggers=11, duration_seconds=560.0, name="run1_ok")
    chosen = choose_completed_thermal(
        [aborted, complete], expected_triggers=11
    )
    assert chosen is not None
    assert chosen.name == "run1_ok"
    assert choose_completed_thermal([aborted], expected_triggers=11) is None
    twelve = _candidate(n_triggers=12, duration_seconds=560.0)
    assert choose_completed_thermal([twelve], expected_triggers=11) is None


def test_pairing_requires_identical_temperature_and_surface_sequence() -> None:
    from bcgstudy.thermal_response import sequences_match

    sim = [(49.3, "1"), (46.3, "5"), (48.3, "3")]
    scan = [(49.3, "1"), (46.3, "5"), (48.3, "3")]
    assert sequences_match(sim, scan) is True
    assert sequences_match(sim, [(49.3, "1"), (48.3, "2"), (45.3, "3")]) is False


def test_trial_summary_sequence_is_read_without_utf8_failure(tmp_path: Path) -> None:
    from bcgstudy.thermal_response import read_trial_sequence

    path = tmp_path / "TrialSummary.csv"
    path.write_bytes(
        b"run_id,trial_number,stimulus_temp,selected_surface\n"
        b"1,1,49.3,1\n"
        b"1,2,46.3,5\n"
        b"1,3,48.3,3\xb0\n"
    )
    assert read_trial_sequence(path) == [(49.3, "1"), (46.3, "5"), (48.3, "3")]


def test_reference_keep_mask_never_depends_on_corrected_outputs() -> None:
    from bcgstudy.thermal_response import reference_keep_mask

    simulator_complete = np.array([True, True, True, False])
    simulator_quality = np.array([True, False, True, True])
    scanner_complete = np.array([True, True, False, True])

    keep = reference_keep_mask(
        simulator_complete,
        simulator_quality,
        scanner_complete,
    )
    np.testing.assert_array_equal(keep, [True, False, False, False])


def test_mark_ecg_excluded_from_eeg_picks() -> None:
    from bcgstudy.thermal_response import eeg_channel_names

    names = ["Fp1", "Cz", "ECG", "Oz", "P3"]
    assert eeg_channel_names(names) == ["Fp1", "Cz", "Oz", "P3"]
    assert "ECG" not in eeg_channel_names(names)


def test_trig_therm_markers_are_returned_as_zero_based_samples(tmp_path: Path) -> None:
    from bcgstudy.thermal_response import trig_therm_samples

    vmrk = tmp_path / "run.vmrk"
    vmrk.write_text(
        "Brain Vision Data Exchange Marker File Version 1.0\n"
        "[Marker Infos]\n"
        "Mk1=Trig_therm,T  1,1,1,0\n"
        "Mk2=Trig_therm,T  1,5100,1,0\n",
        encoding="latin-1",
    )
    assert trig_therm_samples(vmrk) == [0, 5099]


def test_trig_therm_rejects_non_increasing_samples(tmp_path: Path) -> None:
    from bcgstudy.thermal_response import trig_therm_samples

    vmrk = tmp_path / "run.vmrk"
    vmrk.write_text(
        "Brain Vision Data Exchange Marker File Version 1.0\n"
        "[Marker Infos]\n"
        "Mk1=Trig_therm,T  1,100,1,0\n"
        "Mk2=Trig_therm,T  1,100,1,0\n",
        encoding="latin-1",
    )
    with pytest.raises(ValueError, match="non-increasing"):
        trig_therm_samples(vmrk)


def test_add_fastr_improvement_uses_each_participants_own_reference() -> None:
    from bcgstudy.thermal_response import add_fastr_improvement

    rows = [
        {"bids_id": "sub-0019", "method": "fastr", "median_absolute_error": 3.0},
        {"bids_id": "sub-0019", "method": "aas", "median_absolute_error": 1.0},
        {"bids_id": "sub-0020", "method": "fastr", "median_absolute_error": 2.0},
        {"bids_id": "sub-0020", "method": "aas", "median_absolute_error": 2.5},
    ]

    enriched = add_fastr_improvement(rows)
    values = {
        (row["bids_id"], row["method"]): row["improvement_vs_fastr"]
        for row in enriched
    }
    assert values[("sub-0019", "aas")] == pytest.approx(2.0)
    assert values[("sub-0020", "aas")] == pytest.approx(-0.5)
    assert values[("sub-0019", "fastr")] == pytest.approx(0.0)


def test_cardiac_residual_requires_values_for_the_paired_runs() -> None:
    from bcgstudy.thermal_response import cardiac_residual_from_summary

    with pytest.raises(ValueError, match="missing cardiac residual"):
        cardiac_residual_from_summary(
            [],
            bids_id="sub-0019",
            method="aas",
            runs=[1, 2],
            column="local_5_ecg_regressed_ratio_aas",
        )


def test_response_spectrum_figure_is_a_frequency_plot(tmp_path: Path) -> None:
    from bcgstudy.thermal_response import plot_response_spectra

    protocol = _protocol()
    freq = np.arange(1.0, 40.5, 1.0)
    curves = {}
    offsets = (
        ("simulator", 0.0),
        ("fastr", -0.3),
        ("aas", -0.4),
        ("pca_obs", -0.8),
        ("bcgnet", -0.2),
    )
    for key, offset in offsets:
        median = np.full(freq.shape, offset)
        median[(freq >= 8.0) & (freq <= 13.0)] = offset - 1.5
        curves[key] = {
            "freq": freq,
            "median": median,
            "q1": median - 0.2,
            "q3": median + 0.2,
        }
    output = tmp_path / "thermal_response_spectra.png"
    figure = plot_response_spectra(curves, output, protocol)
    assert output.is_file()
    assert output.with_suffix(".pdf").is_file()
    assert len(figure.axes) == 1
    axis = figure.axes[0]
    assert "Frequency" in (axis.get_xlabel() or "")
    assert "dB" in (axis.get_ylabel() or "")
    assert len(axis.lines) >= 5
    caption = " ".join(text.get_text() for text in figure.texts)
    assert "ground truth" not in caption.lower()
    assert "collateral" not in caption.lower()
    assert "FASTR" in caption


def test_prestim_plateau_figure_has_one_spectrum_panel_per_series(
    tmp_path: Path,
) -> None:
    from bcgstudy.thermal_response import plot_prestim_plateau_spectra

    protocol = _protocol()
    freq = np.arange(1.0, 40.5, 1.0)
    panels = {}
    for key in ("simulator", "fastr", "aas", "pca_obs", "bcgnet"):
        prestim = np.linspace(2.0, 0.2, freq.size)
        plateau = prestim * 0.7
        panels[key] = {
            "freq": freq,
            "prestim": prestim,
            "plateau": plateau,
        }
    output = tmp_path / "prestim_plateau_psd.png"
    figure = plot_prestim_plateau_spectra(panels, output, protocol)
    assert output.is_file()
    visible = [axis for axis in figure.axes if axis.get_visible()]
    assert len(visible) == 5
    for axis in visible:
        assert axis.get_yscale() == "log"


def _sine_epochs(amplitude: float, seconds: float, sfreq: float) -> np.ndarray:
    protocol = _protocol()
    n_times = int(seconds * sfreq)
    t = np.arange(n_times) / sfreq
    wave = amplitude * np.sin(2.0 * np.pi * 10.0 * t)
    return np.broadcast_to(
        wave, (1, len(protocol.posterior_channels), n_times)
    ).copy()


def _candidate(*, n_triggers: int, duration_seconds: float, name: str = "run"):
    from bcgstudy.thermal_response import ThermalRecording

    return ThermalRecording(
        bids_id="sub-0019",
        run=1,
        name=name,
        vhdr=Path(f"/tmp/{name}.vhdr"),
        n_trig_therm=n_triggers,
        duration_seconds=duration_seconds,
    )
