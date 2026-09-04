from pathlib import Path

from bcg_correction.bcg_config import _SUPPORTED_CORRECTION_METHODS
from bcgnet.compare.arms import (
    AAS,
    BCGNET,
    CLEAN_ARMS,
    COMPARATOR_ARMS,
    PCA_OBS,
)
from bcgstudy.correction_batch import correction_output_vhdr


def test_every_arm_is_named_after_its_own_method() -> None:
    """No arm carries another arm's name, or a name predating the comparison."""
    src = Path("BaselineEEG_sub0000_fastr.vhdr")
    out = correction_output_vhdr(Path("/aas"), "sub-0000", src, arm=AAS)
    assert out.name == "BaselineEEG_sub0000_fastr_aas.vhdr"


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


def test_comparator_commands_are_derived_from_arm_keys() -> None:
    assert {arm.command for arm in COMPARATOR_ARMS} == {
        "aas",
        "pca-obs",
    }


def test_comparator_arm_keys_are_methods_bcg_correction_accepts() -> None:
    """``arm.key`` is passed straight through as the correction method."""
    for arm in COMPARATOR_ARMS:
        assert arm.key in _SUPPORTED_CORRECTION_METHODS


def test_bcgnet_is_not_a_comparator_arm() -> None:
    assert BCGNET not in COMPARATOR_ARMS
    assert BCGNET in CLEAN_ARMS


def test_every_arm_has_a_distinct_registered_colour() -> None:
    """An arm without a colour falls back, and a fallback that matches another
    line on the same axes is invisible rather than obviously wrong.
    """
    from bcg_correction.figure_style import ARM_COLORS, ARM_FALLBACK, UNCORRECTED
    from bcgnet.compare.arms import CLEAN_ARMS

    missing = [arm.key for arm in CLEAN_ARMS if arm.key not in ARM_COLORS]
    assert not missing, f"arms with no registered colour: {missing}"

    colours = [ARM_COLORS[arm.key] for arm in CLEAN_ARMS]
    assert len(set(colours)) == len(colours), "two arms share a colour"
    assert UNCORRECTED not in colours, "an arm is drawn as the uncorrected trace"
    assert ARM_FALLBACK not in colours
    assert ARM_FALLBACK != UNCORRECTED, "an unregistered arm would look correct"


def test_every_arm_writes_a_distinct_filename_suffix() -> None:
    """Two arms sharing a suffix would overwrite each other on disk."""
    from bcgnet.compare.arms import CLEAN_ARMS

    suffixes = [arm.suffix for arm in CLEAN_ARMS]
    assert len(set(suffixes)) == len(suffixes)
    keys = [arm.key for arm in CLEAN_ARMS]
    assert len(set(keys)) == len(keys)


def test_every_arm_has_a_distinct_registered_marker() -> None:
    """Colour alone does not identify an arm.

    Okabe-Ito separates AAS from PCA-OBS by only dE 7.6 under deuteranopia,
    inside the band that needs a second encoding, so the figures carry a marker
    per arm. An arm with no marker falls back to one shared shape, which reads
    as a registered arm rather than as a mistake.
    """
    from bcg_correction.figure_style import (
        ARM_MARKERS,
        UNCORRECTED_MARKER,
        arm_marker,
    )
    from bcgnet.compare.arms import CLEAN_ARMS

    missing = [arm.key for arm in CLEAN_ARMS if arm.key not in ARM_MARKERS]
    assert not missing, f"arms with no registered marker: {missing}"

    markers = [ARM_MARKERS[arm.key] for arm in CLEAN_ARMS]
    assert len(set(markers)) == len(markers), "two arms share a marker"
    assert UNCORRECTED_MARKER not in markers, "an arm is marked as the input trace"
    assert arm_marker("not-an-arm") not in markers


def test_every_arm_has_a_registered_display_label() -> None:
    """Single-arm pages key by method, so an unregistered arm prints its key."""
    from bcg_correction.figure_style import ARM_LABELS
    from bcgnet.compare.arms import CLEAN_ARMS

    missing = [arm.key for arm in CLEAN_ARMS if arm.key not in ARM_LABELS]
    assert not missing, f"arms with no display label: {missing}"

    labels = [ARM_LABELS[arm.key] for arm in CLEAN_ARMS]
    assert len(set(labels)) == len(labels), "two arms share a label"
    assert labels == [arm.label for arm in CLEAN_ARMS], (
        "figure labels disagree with the arm registry"
    )
