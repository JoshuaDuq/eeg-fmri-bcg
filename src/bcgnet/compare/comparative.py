"""Registry-ordered comparison using the common six-panel renderer."""

from bcg_correction.correction_report import save_profile_report

from .arms import CLEAN_ARMS


def save_comparative_report(profiles_by_arm, *, title, output, coverage=None):
    groups = {
        arm.label: profiles_by_arm[arm.key]
        for arm in CLEAN_ARMS
        if profiles_by_arm.get(arm.key)
    }
    missing = (
        None
        if coverage is None
        else {arm.label: coverage[arm.key] for arm in CLEAN_ARMS if arm.key in coverage}
    )
    return save_profile_report(groups, title=title, output=output, coverage=missing)
