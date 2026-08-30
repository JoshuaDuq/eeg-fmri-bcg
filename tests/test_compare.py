from pathlib import Path

import pytest

from bcgnet.compare.arms import AAS, BCGNET, PCA_OBS
from bcgnet.compare.config import load_compare_config
from bcgnet.compare.pairs import bcgnet_output_vhdr, pair_recordings
from bcgnet.config import ConfigurationError
from bcgnet.export import bcgnet_output_vhdr as export_bcgnet_output_vhdr

_HEADER = "Brain Vision Data Exchange Header File Version 1.0\n" + ("x" * 120)

_ARM_FILENAME = {
    "aas": "BaselineEEG_sub0000_fastr_bcg.vhdr",
    "pca_obs": "BaselineEEG_sub0000_fastr_pcaobs.vhdr",
    "bcgnet": "BaselineEEG_sub0000_fastr_bcgnet.vhdr",
}


def _write_compare_yaml(
    tmp_path: Path,
    *,
    arms: tuple[str, ...],
    names: tuple[str, ...] = ("BaselineEEG_sub0000_fastr.vhdr",),
    naming: str = "",
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
run:
  aas: false
  pca_obs: false
  bcgnet: false
correction:
  window_seconds: [-0.2, 0.7]
  ecg_to_bcg_delay_seconds: 0.21
  aas_neighbor_count: 20
  pca_obs_components: 4
  maximum_residual_ratio: 0.5
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
    config = load_compare_config(
        _write_compare_yaml(tmp_path, arms=("aas", "bcgnet"))
    )
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
    src = (
        tmp_path
        / "fastr"
        / "sub-0001"
        / "ThermalPainEEGFMRI_run2_sub0001_fastr.vhdr"
    )
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
    config = load_compare_config(
        _write_compare_yaml(tmp_path, arms=("bcgnet",))
    )
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
        _write_compare_yaml(
            tmp_path, arms=(), extra="compute:\n  workers: 4\n"
        )
    )

    assert config.compute.workers == 4


def test_compare_config_without_a_compute_block_stays_serial(
    tmp_path: Path,
) -> None:
    """Configs written before this knob existed must keep correcting serially."""
    config = load_compare_config(_write_compare_yaml(tmp_path, arms=()))

    assert config.compute.workers == 1
