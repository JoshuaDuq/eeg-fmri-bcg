"""The correction arms that can be compared against FASTR input.

One record per method keeps the output filename, the CSV column prefix, and the
plot legend in a single place, so adding an arm never means editing a tuple that
some other module also hardcodes.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Arm:
    """One correction method as it appears across outputs, metrics, and plots.

    :param key: config flag and CSV column prefix. For the bounded comparator
        arms this is also the ``method`` string handed to ``bcg_correction``.
    :param label: legend text and the key used in the in-memory trace mapping.
    :param suffix: BrainVision stem suffix written by that arm.
    :param style: matplotlib style string for the PSD overlay.
    :param color: matplotlib color for the epoch overlay.
    """

    key: str
    label: str
    suffix: str
    style: str
    color: str


# ``suffix`` for AAS stays "bcg" so recordings produced before PCA-OBS became a
# separate arm remain discoverable without a rename or a re-run.
AAS = Arm(key="aas", label="AAS", suffix="bcg", style="C2--", color="C2")
PCA_OBS = Arm(
    key="pca_obs", label="PCA-OBS", suffix="pcaobs", style="C5--", color="C5"
)
BCGNET = Arm(
    key="bcgnet", label="BCGNet", suffix="bcgnet", style="C3--", color="C3"
)

#: Bounded methods that ``bcg_correction`` can generate from FASTR input.
COMPARATOR_ARMS: tuple[Arm, ...] = (AAS, PCA_OBS)

#: Every corrected arm an overlay may show, in plotting order.
CLEAN_ARMS: tuple[Arm, ...] = (AAS, PCA_OBS, BCGNET)
