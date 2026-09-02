"""One visual language for every figure this project writes.

Both report modules import from here so a recording page, a cohort page, and a
method-vs-method page cannot drift apart in colour, weight, or labelling.

The palette is Okabe-Ito, which stays distinguishable under deuteranopia and
protanopia -- the common red/green confusions. A green-versus-red comparison is
unreadable for roughly one man in twelve, which is not acceptable in a figure
meant for publication.

Conventions, so a reader learns them once:

- Panel titles are descriptive, left-aligned, and never interpretive; the
  reading guide is one caption at the foot of the page.
- A cohort average is drawn as a line with its interquartile band across
  recordings (and, for residuals, the 5-95% band), never as spaghetti.
- Spectra are averaged across recordings by the median, waveforms by the mean:
  a mean spectrum is owned by whichever recording has the most power.
- Where several recordings are summarised, the summary is median [Q1-Q3] with n.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np

#: Okabe-Ito, chosen for colour-vision deficiency safety.
INK = "#1a1c20"
MUTED = "#6f7681"
FAINT = "#b9bec6"
GRID = "#e9ebef"
UNCORRECTED = "#0072B2"   # blue
CORRECTED = "#D55E00"     # vermillion
ARTIFACT = "#009E73"      # bluish green
COLLATERAL = "#CC79A7"    # reddish purple
REFERENCE = "#6c737d"     # neutral grey

#: Per-arm colours. Distinguishable in greyscale as well by line weight.
#: Okabe-Ito, which is distinguishable under every common colour vision type.
#: Every arm needs an entry here: an arm without one falls back, and a fallback
#: that collides with another line on the same axes is a silent plotting bug.
ARM_COLORS = {
    "aas": "#009E73",           # bluish green
    "pca_obs": "#CC79A7",       # reddish purple
    "blocked_mean": "#E69F00",  # orange
    "bcgnet": "#D55E00",        # vermillion
}
#: Deliberately not any palette colour, and not ``UNCORRECTED``: an unregistered
#: arm must look wrong rather than look like something else.
ARM_FALLBACK = "#000000"

ALPHA_BAND_HZ = (8.0, 13.0)
#: BCG power lives below this. Ticking harmonics past it is clutter, not
#: information, so the comb stops here.
BCG_BAND_MAX_HZ = 16.0
ALPHA_SHADE = "#F0E442"

#: Raster resolution for every page. Vector copies are written beside the
#: cohort-level pages, which are the ones that end up in a manuscript.
DPI = 200

STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "axes.titlesize": 9.5,
    "axes.titlepad": 6.0,
    "axes.titlelocation": "left",
    "axes.titleweight": "medium",
    "axes.labelsize": 8.8,
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.xmargin": 0.01,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": INK,
    "ytick.color": INK,
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "legend.fontsize": 8.0,
    "legend.frameon": True,
    "legend.framealpha": 0.92,
    "legend.edgecolor": "none",
    "legend.fancybox": False,
    "legend.handlelength": 1.8,
    "legend.borderpad": 0.4,
    "legend.labelspacing": 0.35,
    "font.family": "sans-serif",
    "font.sans-serif": [
        "Helvetica Neue", "Helvetica", "Arial", "Liberation Sans", "DejaVu Sans",
    ],
    "font.size": 8.8,
    "mathtext.default": "regular",
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def arm_color(key: str) -> str:
    return ARM_COLORS.get(key, ARM_FALLBACK)


def panel(axis, letter: str, title: str, subtitle: str = "") -> None:
    axis.annotate(
        letter, xy=(0.0, 1.0), xycoords="axes fraction",
        xytext=(-34.0, 6.0), textcoords="offset points",
        fontsize=12.0, fontweight="bold", color=INK, va="bottom", ha="left",
    )
    axis.set_title(title, loc="left", color=INK)
    if subtitle:
        axis.set_title(subtitle, loc="right", color=MUTED, fontsize=8.0)


def harmonics(
    axis, heart_rate_bpm: float, *, up_to_hz: float = BCG_BAND_MAX_HZ
) -> None:
    if not np.isfinite(heart_rate_bpm) or heart_rate_bpm <= 0:
        return
    fundamental = heart_rate_bpm / 60.0
    positions = [
        index * fundamental for index in range(1, int(up_to_hz / fundamental) + 1)
    ]
    axis.vlines(
        positions, 0.955, 1.0, transform=axis.get_xaxis_transform(),
        color=MUTED, lw=0.8, zorder=3, clip_on=False,
    )


def shade_alpha(axis) -> None:
    axis.axvspan(*ALPHA_BAND_HZ, color=ALPHA_SHADE, alpha=0.25, zorder=0, lw=0)


def zero_lines(axis, *, x: bool = True, y: bool = True) -> None:
    if y:
        axis.axhline(0.0, color=FAINT, lw=0.7, zorder=1)
    if x:
        axis.axvline(0.0, color=INK, lw=0.6, ls=(0, (2, 2)), zorder=1)


def quantile_band(
    axis, x: np.ndarray, stack: np.ndarray, color: str, *,
    quantiles: tuple[float, float] = (25.0, 75.0), alpha: float = 0.2,
    label: str | None = None,
) -> None:
    values = np.asarray(stack, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2:
        return
    low, high = np.nanpercentile(values, list(quantiles), axis=0)
    axis.fill_between(x, low, high, color=color, alpha=alpha, lw=0, label=label,
                      zorder=1.5)


def spectrum_summary(stack: np.ndarray) -> np.ndarray:
    values = np.asarray(stack, dtype=float)
    return np.nanmedian(values, axis=0) if values.ndim == 2 else values


def strip(
    axis, groups: list[tuple[str, np.ndarray, str]], *, unit: str = "",
    fmt: str = "{:.2f}", seed: int = 7, percent: bool = False,
) -> None:
    rng = np.random.default_rng(seed)
    scale = 100.0 if percent else 1.0
    for position, (_label, values, color) in enumerate(groups):
        data = np.asarray(values, dtype=float) * scale
        data = data[np.isfinite(data)]
        if data.size == 0:
            continue
        jitter = rng.uniform(-0.16, 0.16, size=data.size)
        axis.scatter(
            position + jitter, data, s=9, color=color, alpha=0.45, lw=0,
            zorder=2, rasterized=False,
        )
        q1, median, q3 = np.percentile(data, [25, 50, 75])
        axis.plot([position, position], [q1, q3], color=color, lw=1.6, zorder=3,
                  solid_capstyle="butt")
        axis.plot([position - 0.3, position + 0.3], [median, median], color=INK,
                  lw=1.4, zorder=4)
        axis.annotate(
            fmt.format(median), xy=(position, q3), xytext=(0, 5),
            textcoords="offset points", ha="center", va="bottom",
            fontsize=7.4, color=INK,
        )
    axis.set_xticks(range(len(groups)))
    axis.set_xticklabels([label for label, _v, _c in groups], rotation=90,
                         fontsize=7.4)
    axis.set_xlim(-0.6, len(groups) - 0.4)
    axis.grid(axis="x", visible=False)
    axis.tick_params(axis="x", length=0)
    if unit:
        axis.set_ylabel(unit)


def robust_ylim(
    axis, *series: np.ndarray, pad: float = 0.12, percentiles=(0.5, 99.5)
) -> None:
    values = np.concatenate([np.asarray(s).ravel() for s in series if s is not None])
    values = values[np.isfinite(values)]
    if values.size == 0:
        return
    low, high = np.percentile(values, list(percentiles))
    if high <= low:
        return
    margin = (high - low) * pad
    axis.set_ylim(low - margin, high + margin)


def full_ylim(axis, *series: np.ndarray, pad: float = 0.08) -> None:
    robust_ylim(axis, *series, pad=pad, percentiles=(0.0, 100.0))


def log_ylim(axis, *series: np.ndarray, decades: float = 4.0) -> None:
    values = np.concatenate([np.asarray(s).ravel() for s in series if s is not None])
    values = values[np.isfinite(values) & (values > 0)]
    if values.size == 0:
        return
    high = float(np.percentile(values, 99.8))
    axis.set_ylim(high / (10.0**decades), high * 2.0)


def legend(axis, loc: str = "upper right", **kwargs):
    return axis.legend(loc=loc, **kwargs)


def figure_caption(figure, text: str, *, width: int = 190) -> None:
    body = "\n".join(textwrap.wrap(" ".join(text.split()), width=width))
    figure.text(
        0.01, -0.01, body, ha="left", va="top", fontsize=7.8, color=MUTED,
        linespacing=1.35,
    )


def save_figure(figure, output: Path, *, vector: bool = False) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=DPI, bbox_inches="tight")
    if vector:
        figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
