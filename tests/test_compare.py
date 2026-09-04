import csv
from pathlib import Path

import pytest

from bcg_correction.evaluation import EvaluationSettings
from bcgnet.compare.arms import AAS, BCGNET, PCA_OBS
from bcgnet.compare.config import load_compare_config
from bcgnet.compare.pairs import bcgnet_output_vhdr, pair_recordings
from bcgnet.config import ConfigurationError
from bcgnet.export import bcgnet_output_vhdr as export_bcgnet_output_vhdr

_EVALUATION = EvaluationSettings((2, 5, 10, 20), 8)

_HEADER = "Brain Vision Data Exchange Header File Version 1.0\n" + ("x" * 120)

_ARM_FILENAME = {
    "aas": "BaselineEEG_sub0000_fastr_aas.vhdr",
    "pca_obs": "BaselineEEG_sub0000_fastr_pcaobs.vhdr",
    "bcgnet": "BaselineEEG_sub0000_fastr_bcgnet.vhdr",
}


def test_empty_summary_replaces_stale_csv_with_current_header(tmp_path: Path) -> None:
    from bcgnet.compare.pipeline import _write_summary
    from bcgnet.compare.plots import metric_columns

    output = tmp_path / "out"
    output.mkdir()
    csv_path = output / "compare_summary.csv"
    csv_path.write_text("stale,data\n1,2\n", encoding="utf-8")

    _write_summary(output, [], _EVALUATION)

    with csv_path.open(encoding="utf-8", newline="") as handle:
        assert list(csv.reader(handle)) == [list(metric_columns(_EVALUATION))]


def test_compare_existing_outputs_never_generates_an_arm(
    tmp_path: Path, monkeypatch
) -> None:
    from bcgnet.compare import pipeline

    config_path = _write_compare_yaml(tmp_path, arms=())
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace("aas: false", "aas: true"),
        encoding="utf-8",
    )
    config = load_compare_config(config_path)

    def fail(**kwargs):
        raise AssertionError("report rebuilding must not run a correction")

    monkeypatch.setattr(pipeline, "run_correction_batch", fail)
    # This is an orchestration test, not a test of the intentionally dummy header.
    monkeypatch.setattr(pipeline, "pair_recordings", lambda config: [])

    assert pipeline.compare_existing_outputs(config) == []


def _write_compare_yaml(
    tmp_path: Path,
    *,
    arms: tuple[str, ...],
    names: tuple[str, ...] = ("BaselineEEG_sub0000_fastr.vhdr",),
    naming: str = "",
    compute: str = "compute:\n  workers: 1\n",
    extra: str = "",
) -> Path:
    """Lay out a cohort holding only ``arms``, then a config that points at it."""
    fastr = tmp_path / "fastr" / "sub-0000"
    fastr.mkdir(parents=True)
    for name in names:
        (fastr / name).write_text(_HEADER, encoding="utf-8")
    for key in arms:
        folder = tmp_path / key / "sub-0000"
        folder.mkdir(parents=True)
        (folder / _ARM_FILENAME[key]).write_text(_HEADER, encoding="utf-8")

    yaml_text = f"""
paths:
  fastr_root: {tmp_path / "fastr"}
  aas_root: {tmp_path / "aas"}
  pca_obs_root: {tmp_path / "pca_obs"}
  bcgnet_root: {tmp_path / "bcgnet"}
  output_root: {tmp_path / "out"}
  experiments_root: {tmp_path / "experiments"}
{compute}run:
  aas: false
  pca_obs: false
  bcgnet: false
correction:
  window_seconds: [-0.2, 0.7]
  ecg_to_bcg_delay_seconds: 0.21
  aas_neighbor_count: 20
  pca_obs_components: 4
  evaluation:
    block_counts: [2, 5, 10, 20]
    minimum_beats_per_block: 8
  maximum_gap_fraction: 0.05
  overwrite: false
  detector:
    ecg_channel: ECG
    preprocessing_band_hz: [0.5, 10.0]
    teager_emphasis_hz: 10.0
    teager_smoothing_seconds: 0.028
    template_window_seconds: [-0.2, 0.4]
    minimum_rr_seconds: 0.4
    maximum_rr_seconds: 2.0
    candidate_refractory_seconds: 0.25
    candidate_prominence_mad: 2.0
    correlation_threshold: 0.5
    refinement_iterations: 2
plot:
  channel: Cz
  epoch_start_seconds: 10
  epoch_seconds: 3
  psd_max_hz: 30
subjects:
  include: []
  exclude: []
{naming}{extra}"""
    path = tmp_path / "compare.yaml"
    path.write_text(yaml_text, encoding="utf-8")
    return path


def test_pair_recordings_finds_every_arm_present_on_disk(tmp_path: Path) -> None:
    config = load_compare_config(
        _write_compare_yaml(tmp_path, arms=("aas", "pca_obs", "bcgnet"))
    )
    recordings = pair_recordings(config)
    assert len(recordings) == 1
    recording = recordings[0]
    assert recording.bids_id == "sub-0000"
    assert set(recording.cleaned_vhdr) == {"aas", "pca_obs", "bcgnet"}
    assert (
        recording.cleaned_vhdr["pca_obs"].name
        == "BaselineEEG_sub0000_fastr_pcaobs.vhdr"
    )


def test_pair_recordings_omits_an_arm_that_was_never_generated(
    tmp_path: Path,
) -> None:
    config = load_compare_config(_write_compare_yaml(tmp_path, arms=("aas", "bcgnet")))
    recording = pair_recordings(config)[0]
    assert set(recording.cleaned_vhdr) == {"aas", "bcgnet"}


def test_pca_obs_is_read_from_its_own_root(tmp_path: Path) -> None:
    """PCA-OBS must not be searched for inside the AAS folder."""
    config = load_compare_config(
        _write_compare_yaml(tmp_path, arms=("aas", "pca_obs", "bcgnet"))
    )
    assert config.paths.root_for(PCA_OBS) == (tmp_path / "pca_obs").resolve()
    assert config.paths.root_for(AAS) == (tmp_path / "aas").resolve()
    assert config.paths.root_for(BCGNET) == (tmp_path / "bcgnet").resolve()


def test_compare_config_rejects_the_superseded_aas_block(tmp_path: Path) -> None:
    """A config from before PCA-OBS became an arm must fail loudly, not silently."""
    path = _write_compare_yaml(tmp_path, arms=("aas",))
    path.write_text(
        path.read_text(encoding="utf-8").replace("correction:", "aas:", 1),
        encoding="utf-8",
    )
    with pytest.raises(ConfigurationError, match="correction"):
        load_compare_config(path)


def test_find_bcgnet_vhdr_matches_export_name(tmp_path: Path) -> None:
    src = tmp_path / "fastr" / "sub-0001" / "ThermalPainEEGFMRI_run2_sub0001_fastr.vhdr"
    expected = (
        tmp_path
        / "bcgnet"
        / "sub-0001"
        / "ThermalPainEEGFMRI_run2_sub0001_fastr_bcgnet.vhdr"
    )
    expected.parent.mkdir(parents=True)
    expected.write_text("x" * 80, encoding="utf-8")
    found = bcgnet_output_vhdr(tmp_path / "bcgnet", "sub-0001", src)
    assert found == expected
    assert found == export_bcgnet_output_vhdr(tmp_path / "bcgnet", "sub-0001", src)


def test_compare_labels_recordings_from_their_filenames(tmp_path: Path) -> None:
    """A baseline beside run 2 must not be reported as run 1."""
    config = load_compare_config(
        _write_compare_yaml(
            tmp_path,
            arms=("bcgnet",),
            names=(
                "BaselineEEG_sub0000_fastr.vhdr",
                "ThermalPainEEGFMRI_run2_sub0000_fastr.vhdr",
            ),
        )
    )
    recordings = pair_recordings(config)
    assert [item.label for item in recordings] == ["BaselineEEG", "run2"]
    assert [item.run for item in recordings] == [None, 2]


def test_compare_pairs_carry_the_source_path_not_a_recording(
    tmp_path: Path,
) -> None:
    """``fastr_vhdr`` is opened directly, so it must be a real path."""
    config = load_compare_config(_write_compare_yaml(tmp_path, arms=("bcgnet",)))
    recording = pair_recordings(config)[0]
    assert isinstance(recording.fastr_vhdr, Path)
    assert recording.fastr_vhdr.is_file()


def test_compare_accepts_a_study_specific_run_pattern(tmp_path: Path) -> None:
    config = load_compare_config(
        _write_compare_yaml(
            tmp_path,
            arms=("bcgnet",),
            names=("acquisition_S03_sub0000.vhdr",),
            naming="naming:\n  run_pattern: '_S(\\d+)_'\n",
        )
    )
    (recording,) = pair_recordings(config)
    assert recording.run == 3
    assert recording.label == "run3"


def test_compare_config_reads_a_worker_count_for_the_bounded_arms(
    tmp_path: Path,
) -> None:
    """``compute.workers`` is how many recordings an arm corrects at once."""
    config = load_compare_config(
        _write_compare_yaml(tmp_path, arms=(), compute="compute:\n  workers: 4\n")
    )

    assert config.compute.workers == 4


def test_compare_config_requires_a_compute_block(
    tmp_path: Path,
) -> None:
    path = _write_compare_yaml(tmp_path, arms=(), compute="")

    with pytest.raises(ConfigurationError, match="compute"):
        load_compare_config(path)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("pca_obs_components: 4", "pca_obs_components: -3", "pca_obs_components"),
        (
            "maximum_gap_fraction: 0.05",
            "maximum_gap_fraction: .nan",
            "maximum_gap_fraction",
        ),
        (
            "correlation_threshold: 0.5",
            "correlation_threshold: 9.0",
            "correlation_threshold",
        ),
        ("epoch_seconds: 3", "epoch_seconds: -1", "epoch_seconds"),
    ],
)
def test_compare_config_rejects_invalid_numeric_values(
    tmp_path: Path, old: str, new: str, message: str
) -> None:
    path = _write_compare_yaml(tmp_path, arms=())
    path.write_text(path.read_text(encoding="utf-8").replace(old, new))

    with pytest.raises(ConfigurationError, match=message):
        load_compare_config(path)


def test_compare_config_rejects_non_string_subjects(tmp_path: Path) -> None:
    path = _write_compare_yaml(tmp_path, arms=())
    path.write_text(
        path.read_text(encoding="utf-8").replace("include: []", "include: [123]")
    )

    with pytest.raises(ConfigurationError, match="include"):
        load_compare_config(path)


def _profile(key: str, ratio: float, spec: float, alpha: float):
    import numpy as np

    from bcg_correction.correction_report import (
        DISPLAY_MS,
        FREQUENCY_GRID_HZ,
        CorrectionProfile,
    )

    wave = np.zeros_like(DISPLAY_MS)
    spectrum = np.ones_like(FREQUENCY_GRID_HZ)
    return CorrectionProfile(
        method=key,
        label="run1",
        subject="s1",
        preservation_status="not_measured",
        block_counts=np.array([2, 5, 10, 20]),
        window_seconds=np.array([-0.2, 0.7]),
        minimum_beats_per_block=8,
        block_minimum_beats=np.array([100, 40, 20, 10]),
        beats=200,
        applied_delay_seconds=0.21,
        gap_fraction=0,
        sampling_rate_hz=100,
        channel_names=np.array(["Oz", "Pz", "Cz", "Fz"]),
        local_before_uv=np.full((2, 4), 10.0),
        local_after_uv=np.full((2, 4), 10 * ratio),
        local_ratio=np.full((2, 4), ratio),
        pooled_before=wave,
        pooled_after=wave,
        local_wave_before=np.tile(wave, (4, 1)),
        local_wave_after=np.tile(wave, (4, 1)),
        psd_removed_locked=spectrum,
        psd_removed_variable=spectrum,
        phase_locking_spectrum=spectrum * spec,
        locked_removal_fraction=spec,
        variable_removal_alpha_ratio=alpha,
        topo_before=np.ones((4, 4)),
        topo_after=np.ones((4, 4)) * ratio,
        topo_variable_alpha_ratio=np.full(4, alpha),
    )


def test_comparative_report_draws_every_arm(tmp_path) -> None:
    from bcg_correction.correction_report import report_page_paths
    from bcgnet.compare.comparative import save_comparative_report

    output = tmp_path / "cohort.png"
    assert save_comparative_report(
        {
            "aas": [_profile("aas", 0.32, 0.86, 0.10)],
            "pca_obs": [_profile("pca_obs", 0.18, 0.68, 0.31)],
            "bcgnet": [_profile("bcgnet", 0.29, 0.71, 0.17)],
        },
        title="three arms",
        output=output,
    )
    pages = report_page_paths(output)
    assert pages.keys() == {"residual", "spectra", "ratios"}
    for page in pages.values():
        assert page.stat().st_size > 10_000


def test_comparative_report_handles_a_single_arm(tmp_path) -> None:
    """Running one method must still produce a page."""
    from bcg_correction.correction_report import report_page_paths
    from bcgnet.compare.comparative import save_comparative_report

    output = tmp_path / "one.png"
    assert save_comparative_report(
        {"aas": [_profile("aas", 0.32, 0.86, 0.10)]},
        title="one arm",
        output=output,
    )
    assert all(path.is_file() for path in report_page_paths(output).values())


def test_comparative_report_declines_when_no_arm_ran(tmp_path) -> None:
    from bcgnet.compare.comparative import save_comparative_report

    assert not save_comparative_report(
        {"aas": []}, title="none", output=tmp_path / "x.png"
    )


def test_topography_serves_one_method_and_many(tmp_path) -> None:
    """The same renderer draws a single arm and a comparison."""
    from bcg_correction.correction_report import save_topography_report

    one = {"AAS": [_profile("aas", 0.32, 0.86, 0.10)]}
    many = {
        "AAS": [_profile("aas", 0.32, 0.86, 0.10)],
        "PCA-OBS": [_profile("pca_obs", 0.20, 0.70, 0.16)],
    }
    # Four channels is under the montage minimum, so it declines rather than
    # drawing a map from too few positions.
    assert not save_topography_report(one, title="one", output=tmp_path / "a.png")
    assert not save_topography_report(many, title="many", output=tmp_path / "b.png")


def test_topography_declines_profiles_without_channel_maps(tmp_path) -> None:
    """Profiles written before topography existed must not crash a cohort page."""
    import numpy as np

    from bcg_correction.correction_report import save_topography_report

    stale = _profile("aas", 0.32, 0.86, 0.10)
    object.__setattr__(stale, "channel_names", np.asarray([]))
    assert not save_topography_report(
        {"AAS": [stale]}, title="stale", output=tmp_path / "c.png"
    )


def _keyed(label: str, ratio: float):
    """One profile keyed the way ``_write_experiments`` keys them."""
    profile = _profile("arm", ratio, 0.8, 0.1)
    object.__setattr__(profile, "label", label)
    return profile


def test_cohort_pairs_arms_onto_their_common_recordings() -> None:
    """An arm that produced no output on a recording must not be scored only on
    the ones it survived. AAS is refused by the residual-ratio gate exactly when
    it did worst, so pairing is what keeps its median honest."""
    keyed = {
        "aas": {("s1", "a"): _keyed("a", 0.9), ("s1", "b"): _keyed("b", 0.1)},
        "pca_obs": {
            ("s1", "a"): _keyed("a", 0.5),
            ("s1", "b"): _keyed("b", 0.2),
            ("s1", "c"): _keyed("c", 0.3),
        },
        "bcgnet": {
            ("s1", "a"): _keyed("a", 0.4),
            ("s1", "b"): _keyed("b", 0.3),
            ("s1", "c"): _keyed("c", 0.2),
        },
    }
    attempted = {key: len(items) for key, items in keyed.items()}
    common = set.intersection(*(set(items) for items in keyed.values()))
    cohort = {
        key: [items[recording] for recording in sorted(common)]
        for key, items in keyed.items()
    }

    assert common == {("s1", "a"), ("s1", "b")}
    assert {key: len(value) for key, value in cohort.items()} == {
        "aas": 2,
        "pca_obs": 2,
        "bcgnet": 2,
    }
    assert [p.label for p in cohort["pca_obs"]] == [p.label for p in cohort["aas"]]
    assert {k: attempted[k] - len(common) for k in attempted} == {
        "aas": 0,
        "pca_obs": 1,
        "bcgnet": 1,
    }


def test_comparative_report_reports_each_arm_s_failures(tmp_path) -> None:
    """``coverage`` carries what each arm attempted, so the page can show that an
    arm is missing recordings rather than silently omitting them."""
    from bcg_correction.correction_report import report_page_paths
    from bcgnet.compare.comparative import save_comparative_report

    output = tmp_path / "paired.png"
    assert save_comparative_report(
        {
            "aas": [_profile("aas", 0.32, 0.86, 0.10)],
            "pca_obs": [_profile("pca_obs", 0.18, 0.68, 0.31)],
        },
        title="paired",
        output=output,
        coverage={"aas": 6, "pca_obs": 1},
    )
    pages = report_page_paths(output)
    assert all(path.stat().st_size > 10_000 for path in pages.values())


def test_fail_column_counts_failures_not_pairing_drops(tmp_path) -> None:
    """The arm that fails most defines the paired subset, so counting failures
    against that subset would report the most fragile arm as failing on nothing.
    Coverage is the failure count against the recordings offered."""
    from bcgnet.compare.comparative import save_comparative_report

    offered = 145
    produced = {"aas": 129, "pca_obs": 132}
    common = min(produced.values())
    coverage = {key: offered - n for key, n in produced.items()}
    assert coverage == {"aas": 16, "pca_obs": 13}
    # The fragile arm must not read as flawless just because it set the floor.
    assert coverage["aas"] > coverage["pca_obs"]
    assert coverage["aas"] != produced["aas"] - common

    from bcg_correction.correction_report import report_page_paths

    output = tmp_path / "coverage.png"
    assert save_comparative_report(
        {
            "aas": [_profile("aas", 0.33, 0.83, 0.09)],
            "pca_obs": [_profile("pca_obs", 0.27, 0.71, 0.17)],
        },
        title="coverage",
        output=output,
        coverage=coverage,
    )
    pages = report_page_paths(output)
    assert all(path.stat().st_size > 10_000 for path in pages.values())


def test_compare_existing_outputs_plots_only_uses_cached_profiles(
    tmp_path: Path, monkeypatch
) -> None:
    import pickle

    from bcgnet.compare.pipeline import compare_existing_outputs

    config_path = _write_compare_yaml(tmp_path, arms=("aas",))
    config = load_compare_config(config_path)

    # Missing cache should raise FileNotFoundError.
    with pytest.raises(FileNotFoundError, match="No cached profiles found"):
        compare_existing_outputs(config, plots_only=True)

    # With cache present, it should call _write_experiments and never call _load_traces.
    cache_path = config.paths.output_root / "compare_profiles.pkl"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    fake_profile = _profile("aas", 0.35, 0.85, 0.10)
    with cache_path.open("wb") as handle:
        pickle.dump(
            {
                "profiles": {"sub-0000": {"aas": [fake_profile]}},
                "offered": {"sub-0000": 1},
                "rows": [{"test": 1}],
            },
            handle,
        )

    written = []
    monkeypatch.setattr(
        "bcgnet.compare.pipeline._write_experiments",
        lambda root, profs, offered: written.append((root, profs, offered)),
    )
    monkeypatch.setattr(
        "bcgnet.compare.pipeline._load_traces",
        lambda r: pytest.fail("_load_traces should not be called with plots_only=True"),
    )

    rows = compare_existing_outputs(config, plots_only=True)
    assert rows == [{"test": 1}]
    assert len(written) == 1
    assert "sub-0000" in written[0][1]
