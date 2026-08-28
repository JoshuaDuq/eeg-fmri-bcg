from pathlib import Path

from bcgnet.aas_batch import aas_output_vhdr
from bcgnet.compare.config import load_compare_config
from bcgnet.compare.pairs import bcgnet_output_vhdr, pair_recordings
from bcgnet.export import bcgnet_output_vhdr as export_bcgnet_output_vhdr


def _write_compare_yaml(tmp_path: Path) -> Path:
    fastr = tmp_path / "fastr" / "sub-0000"
    aas = tmp_path / "aas" / "sub-0000"
    net = tmp_path / "bcgnet" / "sub-0000"
    fastr.mkdir(parents=True)
    aas.mkdir(parents=True)
    net.mkdir(parents=True)
    (fastr / "BaselineEEG_sub0000_fastr.vhdr").write_text(
        "Brain Vision Data Exchange Header File Version 1.0\n" + ("x" * 120),
        encoding="utf-8",
    )
    (aas / "BaselineEEG_sub0000_fastr_bcg.vhdr").write_text(
        "Brain Vision Data Exchange Header File Version 1.0\n" + ("x" * 120),
        encoding="utf-8",
    )
    (net / "BaselineEEG_sub0000_fastr_bcgnet.vhdr").write_text(
        "Brain Vision Data Exchange Header File Version 1.0\n" + ("x" * 120),
        encoding="utf-8",
    )
    yaml_text = f"""
paths:
  fastr_root: {tmp_path / "fastr"}
  aas_root: {tmp_path / "aas"}
  bcgnet_root: {tmp_path / "bcgnet"}
  output_root: {tmp_path / "out"}
run:
  aas: false
  bcgnet: false
aas:
  method: aas
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
"""
    path = tmp_path / "compare.yaml"
    path.write_text(yaml_text, encoding="utf-8")
    return path


def test_pair_recordings_matches_existing_folders(tmp_path: Path) -> None:
    config = load_compare_config(_write_compare_yaml(tmp_path))
    triples = pair_recordings(config)
    assert len(triples) == 1
    triple = triples[0]
    assert triple.bids_id == "sub-0000"
    assert triple.aas_vhdr is not None
    assert triple.bcgnet_vhdr is not None
    assert triple.bcgnet_vhdr.name == "BaselineEEG_sub0000_fastr_bcgnet.vhdr"


def test_aas_output_name_matches_gapfix_layout(tmp_path: Path) -> None:
    src = tmp_path / "BaselineEEG_sub0000_fastr.vhdr"
    out = aas_output_vhdr(tmp_path / "aas", "sub-0000", src)
    assert out.name == "BaselineEEG_sub0000_fastr_bcg.vhdr"


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
