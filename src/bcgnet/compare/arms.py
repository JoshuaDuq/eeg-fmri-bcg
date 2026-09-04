"""The correction arms that can be compared against FASTR input.

One record per method keeps the output filename, the CSV column prefix, and the
plot legend in a single place, so adding an arm never means editing a tuple that
some other module also hardcodes.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Arm:
    key: str
    label: str
    suffix: str

    @property
    def command(self) -> str:
        return self.key.replace("_", "-")


# Every arm is named after its method. AAS wrote "bcg" while it was the only
# bounded arm; outputs from before this change need renaming to be discovered.
AAS = Arm(key="aas", label="AAS", suffix="aas")
PCA_OBS = Arm(key="pca_obs", label="PCA-OBS", suffix="pcaobs")
BCGNET = Arm(key="bcgnet", label="BCGNet", suffix="bcgnet")

#: Bounded methods that ``bcg_correction`` can generate from FASTR input.
COMPARATOR_ARMS: tuple[Arm, ...] = (AAS, PCA_OBS)

#: Every corrected arm an overlay may show, in plotting order.
CLEAN_ARMS: tuple[Arm, ...] = (AAS, PCA_OBS, BCGNET)

ARM_BY_COMMAND = {arm.command: arm for arm in CLEAN_ARMS}
