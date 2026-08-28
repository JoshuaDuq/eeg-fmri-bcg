"""The YAML files shipped in this repository must match the loaders.

These are regression guards, not behaviour tests: a schema change that forgets
to update the study config or the template should fail here rather than at the
start of a cohort run.
"""

from pathlib import Path

import pytest

from bcgnet.compare.arms import AAS, BCGNET, PCA_OBS
from bcgnet.compare.config import load_compare_config
from bcgnet.config import load_config

_REPO = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "relative", ["config.yaml", "examples/config.yaml"]
)
def test_training_configs_load(relative: str) -> None:
    config = load_config(_REPO / relative)
    assert config.preprocess.ecg_channel == "ECG"


@pytest.mark.parametrize(
    "relative", ["compare.yaml", "examples/compare.yaml"]
)
def test_compare_configs_load_with_a_root_for_every_arm(relative: str) -> None:
    config = load_compare_config(_REPO / relative)
    roots = {config.paths.root_for(arm) for arm in (AAS, PCA_OBS, BCGNET)}
    assert len(roots) == 3


def test_study_compare_config_points_at_the_study_training_config() -> None:
    """``bcgnet_config: config.yaml`` must resolve to the sibling study config."""
    text = (_REPO / "compare.yaml").read_text(encoding="utf-8")
    assert "bcgnet_config: config.yaml" in text
    assert (_REPO / "config.yaml").is_file()
