"""Method-vs-method pages: the same six panels, one trace per arm.

``bcg_correction.correction_report`` shows one arm before and after. This shows
every arm that ran, on shared axes, so the arms can be read against each other
rather than against separate figures.

No arm is privileged. The order and colours come from ``arms.CLEAN_ARMS``, so a
new method appears here as soon as it is declared there.

Panel D is the one that decides a comparison. Cardiac harmonics are phase-locked
by definition, so whatever a method removed that is *not* phase-locked cannot be
BCG. An arm sitting high there inside the shaded alpha band is removing neural
signal, and it will do that while posting the *best* residual ratio in panel A --
which is why the two panels have to be read together. Panel F then shows every
recording, not a median, so a difference between arms can be weighed against the
spread within an arm.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.lines import Line2D

from bcg_correction.correction_report import (
    DISPLAY_MS,
    FREQUENCY_GRID_HZ,
    REPORT_MAX_HZ,
    CorrectionProfile,
)
from bcg_correction.figure_style import (
    FAINT,
    INK,
    MUTED,
    STYLE,
    UNCORRECTED,
    arm_color,
    figure_caption,
    full_ylim,
    harmonics,
    log_ylim,
    panel,
    quantile_band,
    save_figure,
    shade_alpha,
    spectrum_summary,
    strip,
    zero_lines,
)

from .arms import CLEAN_ARMS

matplotlib.use("Agg")

_WAVES = ("template_before", "template_after", "removed_locked")
_SPECTRA = (
    "psd_before", "psd_after", "psd_present", "psd_removed_locked",
    "psd_removed_nonlocked",
)




def _figure_legend(figure, present) -> None:
    """One legend for the page: the arms keep one colour in every panel."""
    handles = [Line2D([], [], color=UNCORRECTED, lw=2.0, label="uncorrected")]
    handles += [
        Line2D([], [], color=arm_color(arm.key), lw=1.8, label=arm.label)
        for arm in present
    ]
    handles.append(
        Line2D([], [], color=MUTED, lw=1.4, label="present in recording (D)")
    )
    figure.legend(
        handles=handles, loc="outside upper right", ncols=len(handles),
        frameon=False, fontsize=8.6, handlelength=2.2, columnspacing=1.6,
    )


def _caption(present, profiles_by_arm, coverage) -> str:
    counts = ", ".join(
        f"{arm.label} {len(profiles_by_arm[arm.key])}" for arm in present
    )
    if coverage:
        failures = ", ".join(
            f"{arm.label} {coverage.get(arm.key, 0)}" for arm in present
        )
        pairing = (
            f"Arms are paired: every arm is scored on the same recordings, so no "
            f"arm's summary excludes its own failures. Recordings an arm could "
            f"not correct at all, and which were therefore dropped from every "
            f"arm: {failures}."
        )
    else:
        pairing = (
            "Arms are NOT paired: each arm is scored on the recordings it "
            "survived, so an arm that failed on its hardest recordings is "
            "flattered."
        )
    return (
        f"Recordings per arm: {counts}. {pairing} Posterior EEG, ECG channel "
        "regressed out before every measurement except the ratio marked 'as "
        "written'; time is relative to the ECG R peak (dotted line). Lines are "
        "means (waveforms) or medians (spectra) across recordings; bands are "
        "interquartile ranges. Shaded band: alpha, 8-13 Hz; ticks above a "
        "spectrum mark cardiac harmonics. D and E: whatever an arm removed that "
        "is not heartbeat-locked cannot be BCG, so height in D inside the alpha "
        "band is neural signal lost, and it lowers the ratio in A and F exactly "
        "as removing artifact does. A beat-invariant template scores near zero "
        "collateral by construction, whatever its shape, so a low collateral "
        "alone does not show a method is right. In F, every point is one "
        "recording; the bar is the median and the line the interquartile range. "
        "A gap between the two ratios is ECG-shaped residual left in the file."
    )


def _median(profiles, name: str) -> float:
    values = np.asarray([getattr(p, name) for p in profiles], dtype=float)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else float("nan")


def _time_axis(axis) -> None:
    axis.set_xlabel("time from R peak (ms)")
    axis.set_ylabel("amplitude (µV)")
    axis.set_xlim(DISPLAY_MS[0], DISPLAY_MS[-1])


def _frequency_axis(axis) -> None:
    axis.set_xlabel("frequency (Hz)")
    axis.set_ylabel("PSD (µV²/Hz)")
    axis.set_xlim(0.0, REPORT_MAX_HZ)


def _panel_residual(axis, present, centre, reference) -> None:
    axis.plot(DISPLAY_MS, reference["template_before"], color=UNCORRECTED, lw=2.0)
    for arm in present:
        axis.plot(DISPLAY_MS, centre[arm.key]["template_after"],
                  color=arm_color(arm.key), lw=1.5)
    zero_lines(axis)
    full_ylim(axis, reference["template_before"],
              *[centre[a.key]["template_after"] for a in present])
    panel(axis, "A", "Heartbeat-locked average, uncorrected and after each arm")
    _time_axis(axis)


def _panel_residual_zoom(axis, present, centre, stacks) -> None:
    """A is dominated by the uncorrected artifact; this resolves the residuals."""
    for arm in present:
        quantile_band(axis, DISPLAY_MS, stacks[arm.key]["template_after"],
                      arm_color(arm.key), alpha=0.1)
    for arm in present:
        axis.plot(DISPLAY_MS, centre[arm.key]["template_after"],
                  color=arm_color(arm.key), lw=1.6)
    zero_lines(axis)
    full_ylim(axis, *[centre[a.key]["template_after"] for a in present], pad=0.25)
    panel(axis, "B", "Residual after correction, own scale")
    _time_axis(axis)


def _panel_spectra(axis, present, centre, reference, heart_rate) -> None:
    axis.semilogy(FREQUENCY_GRID_HZ, reference["psd_before"], color=UNCORRECTED,
                  lw=1.6)
    for arm in present:
        axis.semilogy(FREQUENCY_GRID_HZ, centre[arm.key]["psd_after"],
                      color=arm_color(arm.key), lw=1.2)
    harmonics(axis, heart_rate)
    shade_alpha(axis)
    log_ylim(axis, reference["psd_before"],
             *[centre[a.key]["psd_after"] for a in present], decades=3.0)
    panel(axis, "C", "Power spectrum after each arm, posterior mean")
    _frequency_axis(axis)


def _panel_collateral(axis, present, centre, reference, heart_rate) -> None:
    axis.semilogy(FREQUENCY_GRID_HZ, reference["psd_present"], color=MUTED, lw=1.4)
    for arm in present:
        axis.semilogy(FREQUENCY_GRID_HZ, centre[arm.key]["psd_removed_nonlocked"],
                      color=arm_color(arm.key), lw=1.5)
    harmonics(axis, heart_rate)
    shade_alpha(axis)
    log_ylim(axis, reference["psd_present"],
             *[centre[a.key]["psd_removed_nonlocked"] for a in present],
             decades=3.5)
    panel(axis, "D", "Removed power that is not cardiac-locked (collateral)")
    _frequency_axis(axis)


def _panel_spectral_specificity(axis, present, stacks, heart_rate) -> None:
    """At each frequency, what share of the removal was cardiac-locked.

    One curve per arm, bounded in [0, 1], computed per recording and then
    summarised, so the band is the spread across recordings rather than an
    artefact of dividing two cohort means.
    """
    for arm in present:
        locked = stacks[arm.key]["psd_removed_locked"]
        total = locked + stacks[arm.key]["psd_removed_nonlocked"]
        with np.errstate(invalid="ignore", divide="ignore"):
            share = np.where(total > 0, locked / total, np.nan)
        quantile_band(axis, FREQUENCY_GRID_HZ, share, arm_color(arm.key), alpha=0.1)
        axis.plot(FREQUENCY_GRID_HZ, np.nanmedian(share, axis=0),
                  color=arm_color(arm.key), lw=1.6)
    harmonics(axis, heart_rate)
    shade_alpha(axis)
    axis.axhline(0.5, color=FAINT, lw=0.8, ls="--")
    axis.set_ylim(0.0, 1.0)
    axis.set_xlim(0.0, REPORT_MAX_HZ)
    panel(axis, "E", "Cardiac-locked share of removed power")
    axis.set_xlabel("frequency (Hz)")
    axis.set_ylabel("share (1 = pure artifact)")


def _panel_distributions(axis, present, profiles_by_arm) -> None:
    """Every recording, per arm, for the three numbers the comparison turns on."""
    axis.axis("off")
    panel(axis, "F", "Per-recording metrics")
    metrics = (
        ("locked ratio\nECG regressed", "locked_ratio", False, "{:.2f}"),
        ("locked ratio\nas written", "locked_ratio_raw", False, "{:.2f}"),
        ("alpha collateral\n(%)", "alpha_collateral_fraction", True, "{:.1f}"),
    )
    width = 0.27
    gap = (1.0 - width * len(metrics)) / (len(metrics) - 1)
    for index, (label, field, percent, fmt) in enumerate(metrics):
        inset = axis.inset_axes([index * (width + gap), 0.0, width, 0.86])
        groups = [
            (
                arm.label,
                np.asarray(
                    [getattr(p, field) for p in profiles_by_arm[arm.key]], dtype=float
                ),
                arm_color(arm.key),
            )
            for arm in present
        ]
        strip(inset, groups, fmt=fmt, percent=percent)
        inset.set_title(label, fontsize=8.0, color=INK, loc="left", pad=10.0)
        if all(not np.any(np.isfinite(values)) for _l, values, _c in groups):
            inset.text(0.5, 0.5, "not in these\nprofiles;\nrerun\nbcg reports",
                       transform=inset.transAxes, ha="center", va="center",
                       fontsize=7.4, color=MUTED)
        inset.set_ylim(bottom=0.0)
        inset.margins(y=0.16)


def save_comparative_report(
    profiles_by_arm: dict[str, list[CorrectionProfile]],
    *,
    title: str,
    output: Path,
    coverage: dict[str, int] | None = None,
) -> bool:
    import matplotlib.pyplot as plt

    present = [arm for arm in CLEAN_ARMS if profiles_by_arm.get(arm.key)]
    if not present:
        return False
    stacks = {
        arm.key: {
            name: np.vstack([getattr(p, name) for p in profiles_by_arm[arm.key]])
            for name in _WAVES + _SPECTRA
        }
        for arm in present
    }
    centre = {
        arm.key: {
            **{name: np.nanmean(stacks[arm.key][name], axis=0) for name in _WAVES},
            **{name: spectrum_summary(stacks[arm.key][name]) for name in _SPECTRA},
        }
        for arm in present
    }
    reference = centre[present[0].key]
    heart_rate = float(np.nanmedian([
        p.heart_rate_bpm for arm in present for p in profiles_by_arm[arm.key]
    ]))

    with plt.rc_context(STYLE):
        figure, axes = plt.subplots(2, 3, figsize=(15.0, 8.2), layout="constrained")
        _panel_residual(axes[0, 0], present, centre, reference)
        _panel_residual_zoom(axes[0, 1], present, centre, stacks)
        _panel_spectra(axes[0, 2], present, centre, reference, heart_rate)
        _panel_collateral(axes[1, 0], present, centre, reference, heart_rate)
        _panel_spectral_specificity(axes[1, 1], present, stacks, heart_rate)
        _panel_distributions(axes[1, 2], present, profiles_by_arm)
        _figure_legend(figure, present)
        figure.suptitle(title, fontsize=11.5, x=0.01, ha="left", color=INK)
        figure_caption(figure, _caption(present, profiles_by_arm, coverage))
        save_figure(figure, output, vector=True)
        plt.close(figure)
    return True
