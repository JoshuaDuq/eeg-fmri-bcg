"""One visual language for every figure this project writes.

Both report modules import from here so a recording page, a cohort page, and a
method-vs-method page cannot drift apart in colour, weight, or labelling.

The pages are drawn at Nature double-column size (183 mm) with Helvetica, so a
PDF can drop into a manuscript without shrinking the type below 6 pt. Raster
copies are 300 dpi. Colour is Okabe-Ito, which stays distinguishable under
deuteranopia and protanopia — a green-versus-red comparison is unreadable for
roughly one man in twelve.

Conventions, so a reader learns them once:

- Panel titles are descriptive, left-aligned, and never interpretive; a
  qualifier sits under the title, and the reading guide is one caption at the
  foot of the page.
- Marker shape is the arm, line style is the estimator: solid or filled is the
  local/as-written/variable measurement, dashed or open the pooled/ECG-regressed/
  heartbeat-locked one. Colour never carries identity alone — Okabe-Ito separates
  AAS from PCA-OBS by only dE 7.6 under deuteranopia, and PCA-OBS is below 3:1
  against white.
- Continuous traces carry no markers. Markers sit only on discrete points.
- Spectra are averaged across recordings by the median, waveforms by the mean:
  a mean spectrum is owned by whichever recording has the most power. An IQR
  band is the participant distribution, drawn only when three or more
  participants contribute.
- Frequency axes carry 1 Hz minor ticks and a faint harmonic comb: BCG energy
  sits on the cardiac series, and a 1 Hz grid is the honest overlay when heart
  rate is not stored on the profile.
- A distribution panel shows every recording as a point and summarises over
  participants, because recordings repeat within a participant.
- No panel prints a value the axis already carries: no summary cards, no
  medians annotated over a column, no tables.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

#: Okabe-Ito palette, safe for colour-vision deficiency (deuteranopia/protanopia).
INK = "#1A1A1A"
MUTED = "#5C5C5C"
FAINT = "#B5B5B5"
GRID = "#EFEFEF"
UNCORRECTED = "#0072B2"  # blue
CORRECTED = "#D55E00"  # vermillion
ARTIFACT = "#009E73"  # bluish green
COLLATERAL = "#CC79A7"  # reddish purple

#: Per-arm colours. Distinguishable in greyscale as well by line weight.
#: Okabe-Ito, which is distinguishable under every common colour vision type.
#: Every arm needs an entry here: an arm without one falls back, and a fallback
#: that collides with another line on the same axes is a silent plotting bug.
ARM_COLORS = {
    "aas": "#009E73",  # bluish green
    "pca_obs": "#CC79A7",  # reddish purple
    "bcgnet": "#D55E00",  # vermillion
}
#: Deliberately not any palette colour, and not ``UNCORRECTED``: an unregistered
#: arm must look wrong rather than look like something else.
ARM_FALLBACK = "#000000"

#: Marker per arm, so identity never rests on colour alone. Okabe-Ito separates
#: AAS green from PCA-OBS purple by only dE 7.6 under deuteranopia, inside the
#: 6-8 band that is legal only with a secondary encoding, and PCA-OBS sits below
#: 3:1 against white. A shape carries both cases, and survives greyscale print.
#: The page grammar is: marker shape is the arm, line style is the estimator.
ARM_MARKERS = {
    "aas": "o",
    "pca_obs": "s",
    "bcgnet": "^",
}
UNCORRECTED_MARKER = "D"

#: How each arm is written in a legend or an axis label. The comparison path
#: already carries display labels from the arm registry; the single-recording
#: and single-arm pages are keyed by method, and without this they print the
#: internal key.
ARM_LABELS = {
    "aas": "AAS",
    "pca_obs": "PCA-OBS",
    "bcgnet": "BCGNet",
}

ALPHA_BAND_HZ = (8.0, 13.0)
ALPHA_SHADE = "#DDE4EC"

#: Dashed estimator: pooled residual, ECG-regressed sensitivity, locked removal.
DASH = (0, (3.4, 1.65))

#: Nature double-column width (183 mm). Each report is its own page so a
#: spectrum is not squeezed into a sixth of a 2 x 3 grid.
FIGURE_SIZE = (7.24, 8.70)
RESIDUAL_SIZE = (7.24, 3.90)
SPECTRA_SIZE = (7.24, 6.40)
RATIOS_SIZE = (7.24, 6.55)
TOPOGRAPHY_ROW_IN = 1.52

#: Raster resolution for every page. Vector copies sit beside the cohort-level
#: pages, which are the ones that end up in a manuscript. Nature asks for 300 dpi
#: on colour figures; the PDF is the archival file.
DPI = 300

#: Sequential maps for topography. Viridis prints a harsh yellow and collides
#: with the Okabe-Ito greens; these two ramps stay in the blue/orange pair that
#: already identifies the page, and they remain separable under deuteranopia.
RMS_CMAP = LinearSegmentedColormap.from_list(
    "bcg_rms",
    ["#F7F4EE", "#DCE6EF", "#9BB8D0", "#4A7FA8", "#1F4E79", "#0B2438"],
    N=256,
)
ALPHA_CMAP = LinearSegmentedColormap.from_list(
    "bcg_alpha",
    ["#F7F4EE", "#F0D3B8", "#E08B4F", "#B34A12", "#5C1D08"],
    N=256,
)

STYLE = {
    "figure.facecolor": "white",
    "figure.dpi": 150,
    "axes.facecolor": "white",
    "axes.edgecolor": "#404040",
    "axes.labelcolor": INK,
    "axes.titlesize": 7.5,
    "axes.titlepad": 5.0,
    "axes.titlelocation": "left",
    "axes.titleweight": "bold",
    "axes.labelsize": 7.0,
    "axes.labelweight": "normal",
    "axes.linewidth": 0.6,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.xmargin": 0.02,
    "axes.ymargin": 0.03,
    "grid.color": GRID,
    "grid.linewidth": 0.4,
    "grid.alpha": 1.0,
    "xtick.color": INK,
    "ytick.color": INK,
    "xtick.labelsize": 6.0,
    "ytick.labelsize": 6.0,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 2.8,
    "ytick.major.size": 2.8,
    "xtick.minor.size": 1.6,
    "ytick.minor.size": 1.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.width": 0.45,
    "ytick.minor.width": 0.45,
    "xtick.major.pad": 2.2,
    "ytick.major.pad": 2.2,
    "legend.fontsize": 6.5,
    "legend.frameon": False,
    "legend.handlelength": 1.7,
    "legend.handleheight": 0.6,
    "legend.borderpad": 0.25,
    "legend.labelspacing": 0.28,
    "legend.columnspacing": 1.1,
    "font.family": "sans-serif",
    "font.sans-serif": [
        "Helvetica Neue",
        "Helvetica",
        "Arial",
        "Liberation Sans",
        "DejaVu Sans",
    ],
    "font.size": 7.0,
    "mathtext.fontset": "dejavusans",
    "mathtext.default": "regular",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.facecolor": "white",
    "savefig.dpi": DPI,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
}


def arm_color(key: str) -> str:
    return ARM_COLORS.get(key, ARM_FALLBACK)


def arm_label(key: str) -> str:
    return ARM_LABELS.get(key, key)


def arm_marker(key: str) -> str:
    return ARM_MARKERS.get(key, "*")


def panel(axis, letter: str, title: str, subtitle: str = "") -> None:
    """Panel letter, title, and an optional qualifier stacked under the title.

    The qualifier sits under the title rather than opposite it: a right-aligned
    subtitle collided with the title on the narrower panels.
    """
    axis.annotate(
        letter,
        xy=(0.0, 1.0),
        xycoords="axes fraction",
        xytext=(-11.0, 7.0 if subtitle else 4.0),
        textcoords="offset points",
        fontsize=8.5,
        fontweight="bold",
        color=INK,
        va="bottom",
        ha="right",
    )
    axis.set_title(
        title,
        loc="left",
        color=INK,
        fontsize=7.4,
        fontweight="bold",
        pad=11.0 if subtitle else 4.0,
    )
    if subtitle:
        axis.annotate(
            subtitle,
            xy=(0.0, 1.0),
            xycoords="axes fraction",
            xytext=(0.0, 2.5),
            textcoords="offset points",
            fontsize=5.8,
            color=MUTED,
            va="bottom",
            ha="left",
        )


def shade_alpha(axis, *, label: bool = True) -> None:
    axis.axvspan(*ALPHA_BAND_HZ, color=ALPHA_SHADE, alpha=0.65, zorder=0, lw=0)
    if label:
        axis.annotate(
            "8-13 Hz",
            xy=(10.5, 0.97),
            xycoords=("data", "axes fraction"),
            ha="center",
            va="top",
            fontsize=5.8,
            color=MUTED,
            zorder=3,
        )


def harmonic_comb(axis, *, f0: float = 1.0, fmax: float = 24.0) -> None:
    """Faint 1 Hz lines through the BCG harmonic range.

    Drawn as a LineCollection, not Line2D, so it cannot become ``axes.lines[0]``
    and steal a spectrum assertion. The 1 Hz spacing is the conventional overlay
    when the profile does not store heart rate.
    """
    freqs = np.arange(f0, fmax + 0.01, f0)
    axis.vlines(
        freqs,
        0.0,
        1.0,
        transform=axis.get_xaxis_transform(),
        colors="#D4D4D4",
        lw=0.4,
        zorder=0,
        clip_on=True,
    )


def frequency_axis(axis, *, max_hz: float = 45.0) -> None:
    """Publication frequency axis: 5 Hz majors, 1 Hz minors, harmonic comb."""
    axis.set_xlim(1.0, max_hz)
    axis.set_xlabel("Frequency (Hz)")
    majors = np.arange(5.0, max_hz + 0.01, 5.0)
    axis.set_xticks(majors)
    axis.set_xticks(np.arange(1.0, max_hz + 0.01, 1.0), minor=True)
    axis.tick_params(axis="x", which="minor", length=1.6, width=0.4)
    axis.grid(axis="x", visible=False)
    harmonic_comb(axis, fmax=min(24.0, max_hz))
    shade_alpha(axis)


def fill_iqr(axis, x, q1, q3, color, *, alpha: float = 0.14, zorder: float = 1.5):
    """Participant IQR as a band. Silent when the band is undefined."""
    if q1 is None or q3 is None:
        return None
    x = np.asarray(x, dtype=float)
    q1 = np.asarray(q1, dtype=float)
    q3 = np.asarray(q3, dtype=float)
    finite = np.isfinite(x) & np.isfinite(q1) & np.isfinite(q3)
    if np.count_nonzero(finite) < 2:
        return None
    return axis.fill_between(
        x,
        q1,
        q3,
        where=finite,
        interpolate=True,
        color=color,
        alpha=alpha,
        linewidth=0,
        zorder=zorder,
        clip_on=True,
    )


def strip(
    axis,
    groups: list[tuple],
    *,
    unit: str = "",
    seed: int = 7,
    percent: bool = False,
) -> None:
    """One dot per recording, with a median and IQR bar over participants.

    The two arrays carry different units of analysis on purpose. ``points`` is
    every recording, so nothing is hidden behind a summary. ``summary`` is the
    per-participant values, and the bar is drawn from those: recordings repeat
    within a participant, so an IQR taken over recordings would read as far
    tighter than the cohort actually is.

    Each group is ``(label, points, summary, color)`` or
    ``(label, points, summary, color, marker)``. Marker shape is the arm.

    No value is printed on the axes. The scale carries the numbers.
    """
    rng = np.random.default_rng(seed)
    scale = 100.0 if percent else 1.0
    for position, group in enumerate(groups):
        _label, points, summary, color = group[:4]
        marker = group[4] if len(group) > 4 else "o"
        data = np.asarray(points, dtype=float).ravel() * scale
        data = data[np.isfinite(data)]
        if data.size:
            jitter = (
                rng.uniform(-0.16, 0.16, size=data.size)
                if data.size > 1
                else np.zeros(1)
            )
            axis.scatter(
                position + jitter,
                data,
                s=18 if data.size > 1 else 28,
                color=color,
                marker=marker,
                alpha=0.55 if data.size > 1 else 0.9,
                linewidths=0.35,
                edgecolors="white",
                zorder=2,
            )
        centre = np.asarray(summary, dtype=float).ravel() * scale
        centre = centre[np.isfinite(centre)]
        # A median and an IQR over one or two participants describe nothing, and
        # a bar drawn over a single point hides it. Below three, show the points
        # and draw no summary.
        if centre.size < 3:
            continue
        q1, median, q3 = np.percentile(centre, [25, 50, 75])
        axis.plot(
            [position, position],
            [q1, q3],
            color=INK,
            lw=1.15,
            zorder=3,
            solid_capstyle="butt",
        )
        axis.plot(
            [position - 0.18, position + 0.18],
            [median, median],
            color=INK,
            lw=1.7,
            zorder=4,
            solid_capstyle="butt",
        )
    axis.set_xticks(range(len(groups)))
    axis.set_xticklabels([group[0] for group in groups], fontsize=6.5)
    axis.set_xlim(-0.55, len(groups) - 0.45)
    axis.grid(axis="x", visible=False)
    axis.tick_params(axis="x", length=0)
    if unit:
        axis.set_ylabel(unit)


def linestyle_key(
    axis,
    entries: list[tuple[str | tuple, str]],
    *,
    loc: str = "upper right",
):
    """A legend for line style alone, drawn in ink so it cannot read as an arm."""
    from matplotlib.lines import Line2D

    handles = [
        Line2D([], [], color=MUTED, lw=1.15, ls=style, label=label)
        for style, label in entries
    ]
    return axis.legend(
        handles=handles,
        loc=loc,
        fontsize=5.8,
        handlelength=2.0,
        labelspacing=0.22,
        borderpad=0.2,
        frameon=False,
    )


def arm_legend(
    figure,
    entries: list[tuple[str, str, str]],
    *,
    columns: int | None = None,
):
    """One legend for the whole page, so panels do not each repeat the arms."""
    from matplotlib.lines import Line2D

    handles = [
        Line2D(
            [],
            [],
            color=color,
            lw=1.4,
            label=label,
            marker=marker,
            markersize=4.4,
            markeredgecolor="white",
            markeredgewidth=0.4,
        )
        for label, color, marker in entries
    ]
    return figure.legend(
        handles=handles,
        loc="outside upper right",
        ncol=columns or len(handles),
        fontsize=7.0,
        frameon=False,
        handlelength=1.4,
        columnspacing=1.25,
        borderaxespad=0.2,
    )


def figure_caption(figure, text: str, *, width: int = 118) -> None:
    body = "\n".join(textwrap.wrap(" ".join(text.split()), width=width))
    figure.text(
        0.0,
        -0.012,
        body,
        ha="left",
        va="top",
        fontsize=5.6,
        color=MUTED,
        linespacing=1.32,
    )


def save_figure(figure, output: Path, *, vector: bool = False) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=DPI, bbox_inches="tight", pad_inches=0.05)
    if vector:
        figure.savefig(
            output.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.05
        )
