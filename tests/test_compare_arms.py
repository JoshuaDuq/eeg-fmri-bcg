from pathlib import Path

from bcg_correction.bcg_config import _SUPPORTED_CORRECTION_METHODS
from bcgnet.compare.arms import (
    AAS,
    BCGNET,
    CLEAN_ARMS,
    COMPARATOR_ARMS,
    PCA_OBS,
)
from bcgnet.correction_batch import correction_output_vhdr


def test_aas_output_name_stays_on_the_existing_gapfix_layout() -> None:
    """Already-generated ``*_fastr_bcg.vhdr`` files must remain discoverable."""
    src = Path("BaselineEEG_sub0000_fastr.vhdr")
    out = correction_output_vhdr(Path("/aas"), "sub-0000", src, arm=AAS)
    assert out.name == "BaselineEEG_sub0000_fastr_bcg.vhdr"


def test_pca_obs_output_name_is_tagged_with_its_own_method() -> None:
    src = Path("BaselineEEG_sub0000_fastr.vhdr")
    out = correction_output_vhdr(Path("/pca"), "sub-0000", src, arm=PCA_OBS)
    assert out.name == "BaselineEEG_sub0000_fastr_pcaobs.vhdr"


def test_output_is_written_under_the_subject_folder() -> None:
    src = Path("BaselineEEG_sub0000_fastr.vhdr")
    out = correction_output_vhdr(Path("/pca"), "sub-0000", src, arm=PCA_OBS)
    assert out.parent == Path("/pca/sub-0000")


def test_arms_never_collide_on_a_filename_suffix() -> None:
    suffixes = [arm.suffix for arm in CLEAN_ARMS]
    assert len(set(suffixes)) == len(suffixes)


def test_comparator_arm_keys_are_methods_bcg_correction_accepts() -> None:
    """``arm.key`` is passed straight through as the correction method."""
    for arm in COMPARATOR_ARMS:
        assert arm.key in _SUPPORTED_CORRECTION_METHODS


def test_bcgnet_is_not_a_comparator_arm() -> None:
    assert BCGNET not in COMPARATOR_ARMS
    assert BCGNET in CLEAN_ARMS
