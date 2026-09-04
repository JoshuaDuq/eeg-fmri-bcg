"""One measurement profile and three report pages for every correction arm.

Residual waveforms, removal spectra, and residual ratios are separate figures
so a spectrum is not squeezed into a sixth of a grid. All non-ECG input
channels are EEG in this pipeline. The primary view measures the saved EEG
(recording-level median removed); ECG regression is a sensitivity analysis,
not a uniquely identified separation of cardiac and neural signals.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path

import matplotlib
import numpy as np

from .evaluation import (
    EvaluationSettings,
    band_integral,
    divide_or_nan,
    epoch_spectrum,
    local_locked_energy,
    spectral_locked_fraction,
)
from .figure_style import (
    ALPHA_CMAP,
    DASH,
    FAINT,
    INK,
    MUTED,
    RATIOS_SIZE,
    RESIDUAL_SIZE,
    RMS_CMAP,
    SPECTRA_SIZE,
    STYLE,
    TOPOGRAPHY_ROW_IN,
    UNCORRECTED,
    UNCORRECTED_MARKER,
    arm_color,
    arm_label,
    arm_legend,
    arm_marker,
    figure_caption,
    fill_iqr,
    frequency_axis,
    linestyle_key,
    panel,
    save_figure,
    strip,
)
from .metrics import regress_out_reference

DISPLAY_MS = np.arange(-200.0, 901.0)
REPORT_MAX_HZ = 45.0
FREQUENCY_GRID_HZ = np.arange(1.0, REPORT_MAX_HZ + 0.001, 0.25)
REPORT_PAGES = ("residual", "spectra", "ratios")
PROFILE_SCHEMA_VERSION = 2
_TEXT_FIELDS = {"method", "label", "subject", "preservation_status"}
_INT_FIELDS = {"beats", "minimum_beats_per_block"}
_FLOAT_FIELDS = {
    "applied_delay_seconds",
    "gap_fraction",
    "sampling_rate_hz",
    "locked_removal_fraction",
    "variable_removal_alpha_ratio",
}


@dataclass(frozen=True, slots=True)
class CorrectionProfile:
    method: str
    label: str
    subject: str
    preservation_status: str
    block_counts: np.ndarray
    window_seconds: np.ndarray
    minimum_beats_per_block: int
    block_minimum_beats: np.ndarray
    beats: int
    applied_delay_seconds: float
    gap_fraction: float
    sampling_rate_hz: float
    channel_names: np.ndarray
    # Variant x resolution: 0 = as written, 1 = ECG regressed.
    local_before_uv: np.ndarray
    local_after_uv: np.ndarray
    local_ratio: np.ndarray
    pooled_before: np.ndarray
    pooled_after: np.ndarray
    local_wave_before: np.ndarray
    local_wave_after: np.ndarray
    psd_removed_locked: np.ndarray
    psd_removed_variable: np.ndarray
    phase_locking_spectrum: np.ndarray
    locked_removal_fraction: float
    variable_removal_alpha_ratio: float
    topo_before: np.ndarray
    topo_after: np.ndarray
    topo_variable_alpha_ratio: np.ndarray


def _validate_pair(before, after, names, ecg, peaks, fs, window):
    if before.ndim != 2 or before.shape != after.shape:
        raise ValueError("before and after must have the same channel x sample shape")
    if len(names) != before.shape[0] or len(set(names)) != len(names):
        raise ValueError("channel names must be unique and match the data")
    if not 0 <= ecg < len(names) or len(names) < 2:
        raise ValueError("an ECG channel and at least one EEG channel are required")
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("sampling rate must be finite and positive")
    if not np.all(np.isfinite(before)) or not np.all(np.isfinite(after)):
        raise ValueError("EEG data must be finite")
    if (
        peaks.ndim != 1
        or not np.issubdtype(peaks.dtype, np.integer)
        or np.any(np.diff(peaks) <= 0)
        or np.any(peaks < 0)
        or np.any(peaks >= before.shape[1])
    ):
        raise ValueError("R peaks must be increasing integer sample positions")
    if len(window) != 2 or not np.all(np.isfinite(window)) or window[0] >= window[1]:
        raise ValueError("measurement window must be finite and increasing")


def _grid(times, values):
    return np.interp(DISPLAY_MS, times, values, left=np.nan, right=np.nan)


def report_page_paths(output: Path) -> dict[str, Path]:
    """Residual, spectra, and ratio pages derived from one report path."""
    return {
        page: output.with_name(f"{output.stem}_{page}{output.suffix}")
        for page in REPORT_PAGES
    }


def anchor_grid_ms(window_seconds) -> np.ndarray:
    """1 ms grid covering the delay-centred measurement window."""
    lo = float(window_seconds[0]) * 1000.0
    hi = float(window_seconds[1]) * 1000.0
    return np.arange(np.round(lo), np.round(hi) + 0.001, 1.0)


def regrid_to_anchor(values, delay_seconds, grid) -> np.ndarray:
    """Shift an R-peak-time series onto time from the BCG anchor.

    Profiles store waves on ``DISPLAY_MS`` (time from R-peak). Cohort averages
    on that axis smear recordings whose ECG-to-BCG delays differ. The
    measurement window is delay-centred, so interpolating onto
    ``t_R - delay`` aligns every recording's BCG peak at 0 ms.
    """
    y = np.asarray(values, dtype=float)
    grid = np.asarray(grid, dtype=float)
    if y.ndim == 2:
        return np.stack(
            [regrid_to_anchor(row, delay_seconds, grid) for row in y]
        )
    t_anchor = DISPLAY_MS - float(delay_seconds) * 1000.0
    out = np.full(grid.shape, np.nan)
    finite = np.isfinite(y) & np.isfinite(t_anchor)
    if np.count_nonzero(finite) < 2:
        return out
    order = np.argsort(t_anchor[finite])
    t = t_anchor[finite][order]
    v = y[finite][order]
    inside = (grid >= t[0]) & (grid <= t[-1])
    if np.any(inside):
        out[inside] = np.interp(grid[inside], t, v)
    return out


def _spectrum_grid(frequency, power):
    return np.interp(
        FREQUENCY_GRID_HZ,
        frequency,
        np.median(power, axis=0),
        left=np.nan,
        right=np.nan,
    )


def compute_correction_profile(
    before_volts,
    after_volts,
    channel_names,
    *,
    ecg_channel_index,
    peak_samples,
    sampling_rate_hz,
    delay_seconds,
    window_seconds,
    gap_fraction,
    method,
    evaluation: EvaluationSettings,
    label="",
    subject="",
) -> CorrectionProfile | None:
    before = np.asarray(before_volts, dtype=float)
    after = np.asarray(after_volts, dtype=float)
    peaks = np.asarray(peak_samples)
    fs = float(sampling_rate_hz)
    _validate_pair(
        before, after, channel_names, ecg_channel_index, peaks, fs, window_seconds
    )
    if not np.isfinite(delay_seconds) or not 0 <= gap_fraction <= 1:
        raise ValueError("delay and gap fraction must be finite and valid")
    start = round((delay_seconds + window_seconds[0]) * fs)
    stop = round((delay_seconds + window_seconds[1]) * fs)
    if stop - start < 2:
        raise ValueError("measurement window contains fewer than two samples")
    starts = peaks + start
    starts = starts[(starts >= 0) & (starts + stop - start <= before.shape[1])]
    if starts.size < 2:
        return None
    index = starts[:, None] + np.arange(stop - start)
    times = np.arange(start, stop) / fs * 1000
    picks = np.array([i for i in range(len(channel_names)) if i != ecg_channel_index])
    counts = np.asarray(evaluation.block_counts)
    minimum_beats = starts.size // counts
    shape = (2, len(counts))
    before_rms, after_rms = np.full(shape, np.nan), np.full(shape, np.nan)
    maps_before = np.full((len(counts), len(picks)), np.nan)
    maps_after = np.full_like(maps_before, np.nan)
    waves_before = np.full((len(counts), len(DISPLAY_MS)), np.nan)
    waves_after = np.full_like(waves_before, np.nan)
    direct_epochs = []
    pooled = []
    for variant in range(2):
        for data, amplitudes, maps, waves in (
            (before, before_rms, maps_before, waves_before),
            (after, after_rms, maps_after, waves_after),
        ):
            eeg = data[picks]
            if variant == 1:
                # The SAME original ECG reference is used for both signals.
                eeg = regress_out_reference(eeg, before[ecg_channel_index])
            eeg = (eeg - np.median(eeg, axis=1, keepdims=True)) * 1e6
            epochs = eeg[:, index]
            if variant == 0:
                direct_epochs.append(epochs)
                pooled.append(
                    _grid(times, np.sqrt(np.mean(epochs.mean(axis=1) ** 2, axis=0)))
                )
            for resolution, count in enumerate(counts):
                if minimum_beats[resolution] < evaluation.minimum_beats_per_block:
                    continue
                energy = local_locked_energy(epochs, int(count))
                channel_rms = np.sqrt(energy.mean(axis=-1))
                amplitudes[variant, resolution] = np.median(channel_rms)
                if variant == 0:
                    maps[resolution] = channel_rms
                    waves[resolution] = _grid(times, np.sqrt(energy.mean(axis=0)))
    original, corrected = direct_epochs
    removed = original - corrected
    locked = removed.mean(axis=1, keepdims=True)
    variable = removed - locked
    frequency, original_psd = epoch_spectrum(original, fs)
    _, locked_psd = epoch_spectrum(locked, fs)
    _, variable_psd = epoch_spectrum(variable, fs)
    alpha_before = band_integral(frequency, original_psd, 8, 13)
    alpha_variable = band_integral(frequency, variable_psd, 8, 13)
    return CorrectionProfile(
        method=method,
        label=label,
        subject=subject,
        preservation_status="not_measured",
        block_counts=counts,
        window_seconds=np.asarray(window_seconds, dtype=float),
        minimum_beats_per_block=evaluation.minimum_beats_per_block,
        block_minimum_beats=minimum_beats,
        beats=int(starts.size),
        applied_delay_seconds=float(delay_seconds),
        gap_fraction=float(gap_fraction),
        sampling_rate_hz=fs,
        channel_names=np.asarray([channel_names[i] for i in picks]),
        local_before_uv=before_rms,
        local_after_uv=after_rms,
        local_ratio=divide_or_nan(after_rms, before_rms),
        pooled_before=pooled[0],
        pooled_after=pooled[1],
        local_wave_before=waves_before,
        local_wave_after=waves_after,
        psd_removed_locked=_spectrum_grid(frequency, locked_psd),
        psd_removed_variable=_spectrum_grid(frequency, variable_psd),
        phase_locking_spectrum=np.interp(
            FREQUENCY_GRID_HZ,
            frequency,
            spectral_locked_fraction(locked_psd, variable_psd),
            left=np.nan,
            right=np.nan,
        ),
        locked_removal_fraction=float(
            divide_or_nan(np.mean(locked**2), np.mean(removed**2))
        ),
        variable_removal_alpha_ratio=float(
            divide_or_nan(np.median(alpha_variable), np.median(alpha_before))
        ),
        topo_before=maps_before,
        topo_after=maps_after,
        topo_variable_alpha_ratio=divide_or_nan(alpha_variable, alpha_before),
    )


def write_profile(profile: CorrectionProfile, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, schema_version=np.asarray(PROFILE_SCHEMA_VERSION), **asdict(profile)
    )


def profile_metrics(profile: CorrectionProfile) -> dict:
    """Flat, JSON-safe measurements shared by provenance and tabular exports."""
    result = {
        "beats": profile.beats,
        "gap_fraction": profile.gap_fraction,
        "preservation_status": profile.preservation_status,
    }
    for name in ("locked_removal_fraction", "variable_removal_alpha_ratio"):
        value = getattr(profile, name)
        result[name] = float(value) if np.isfinite(value) else None
    for resolution, count in enumerate(profile.block_counts):
        result[f"local_{count}_minimum_beats"] = int(
            profile.block_minimum_beats[resolution]
        )
        for variant, label in enumerate(("as_written", "ecg_regressed")):
            for name in ("before_uv", "after_uv", "ratio"):
                value = getattr(profile, f"local_{name}")[variant, resolution]
                result[f"local_{count}_{label}_{name}"] = (
                    float(value) if np.isfinite(value) else None
                )
    return result


def read_profile(path: Path) -> CorrectionProfile:
    with np.load(path, allow_pickle=False) as data:
        expected = {field.name for field in fields(CorrectionProfile)} | {
            "schema_version"
        }
        if set(data.files) != expected:
            raise ValueError(
                f"invalid profile schema (including schema_version); "
                f"rebuild with bcg reports: {path}"
            )
        if int(data["schema_version"]) != PROFILE_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported profile schema_version; run bcg reports: {path}"
            )
        values = {}
        for field in fields(CorrectionProfile):
            name, value = field.name, data[field.name]
            if name in _TEXT_FIELDS:
                values[name] = str(value)
            elif name in _INT_FIELDS:
                values[name] = int(value)
            elif name in _FLOAT_FIELDS:
                values[name] = float(value)
            else:
                values[name] = np.asarray(value)
    return CorrectionProfile(**values)


def participant_values(profiles, field):
    """Mean within participant, retaining missing scales rather than dropping runs."""
    groups = {}
    for profile in profiles:
        groups.setdefault(profile.subject, []).append(getattr(profile, field))
    if len(profiles) > 1 and "" in groups:
        raise ValueError("subject IDs are required for aggregate reports")
    return np.stack([np.mean(groups[key], axis=0) for key in sorted(groups)])


def _centre(profiles, field):
    return np.median(participant_values(profiles, field), axis=0)


def _band(profiles, field):
    """Participant median and, when n >= 3, the IQR. Otherwise no band."""
    values = participant_values(profiles, field)
    median = np.median(values, axis=0)
    if values.shape[0] < 3:
        return median, None, None
    q1, q3 = np.percentile(values, [25, 75], axis=0)
    return median, q1, q3


def _anchor_values(profiles, field, resolution=None):
    grid = anchor_grid_ms(profiles[0].window_seconds)
    grouped = {}
    for profile in profiles:
        aligned = regrid_to_anchor(
            getattr(profile, field), profile.applied_delay_seconds, grid
        )
        if resolution is not None:
            aligned = aligned[resolution]
        grouped.setdefault(profile.subject, []).append(aligned)
    if len(profiles) > 1 and "" in grouped:
        raise ValueError("subject IDs are required for aggregate reports")
    return grid, np.stack([np.mean(grouped[key], axis=0) for key in sorted(grouped)])


def _anchor_band(profiles, field, resolution=None):
    grid, values = _anchor_values(profiles, field, resolution)
    median = np.median(values, axis=0)
    if values.shape[0] < 3:
        return grid, median, None, None
    q1, q3 = np.percentile(values, [25, 75], axis=0)
    return grid, median, q1, q3


def _positive_log_limits(axis, *, x_lo=1.0, x_hi=25.0, floor=1e-3, decades=4):
    """Log limits from the BCG-relevant band, not the Nyquist tail."""
    samples = []
    for line in axis.get_lines():
        x = np.asarray(line.get_xdata(), dtype=float)
        y = np.asarray(line.get_ydata(), dtype=float)
        keep = (
            np.isfinite(x)
            & np.isfinite(y)
            & (y > 0)
            & (x >= x_lo)
            & (x <= x_hi)
        )
        if np.any(keep):
            samples.append(y[keep])
    y = np.concatenate(samples) if samples else np.array([])
    if y.size == 0:
        return floor, 1.0
    lo = float(np.min(y))
    hi = float(np.max(y))
    ymax = 10 ** np.ceil(np.log10(hi) + 0.08)
    ymin = 10 ** np.floor(np.log10(max(lo, floor)) - 0.05)
    ymin = max(ymin, floor, ymax / (10 ** decades))
    if ymax <= ymin:
        ymax = ymin * 10
    return ymin, ymax


def _validate_groups(groups):
    profiles = [p for items in groups.values() for p in items]
    reference = profiles[0]
    for profile in profiles:
        if (
            not np.array_equal(profile.block_counts, reference.block_counts)
            or not np.array_equal(profile.window_seconds, reference.window_seconds)
            or profile.minimum_beats_per_block != reference.minimum_beats_per_block
            or not np.array_equal(profile.channel_names, reference.channel_names)
        ):
            raise ValueError(
                "profiles have incompatible evaluation settings or channels"
            )
    keys = [set((p.subject, p.label) for p in items) for items in groups.values()]
    if any(
        len(items) != len(key) for items, key in zip(groups.values(), keys, strict=True)
    ):
        raise ValueError("duplicate recording profile")
    if any(key != keys[0] for key in keys[1:]):
        raise ValueError("comparison profiles must use the same paired recordings")


_SHARED = (
    "Descriptive evaluation of saved outputs, not independent validation. All EEG "
    "channels; recording median removed. Shaded bands are participant IQR. No "
    "signal-preservation or overall winner claim is supported."
)
_RESIDUAL_CAPTION = (
    "Time is relative to each recording's estimated ECG-to-BCG delay (BCG anchor), "
    "so cohort averages are not smeared across different R-to-BCG latencies. A "
    "delay of 0 ms is the estimator floor. Solid = local chronological blocks; "
    "dashed = pooled over all beats (cancellation-prone). Local curves square "
    "within blocks first."
)
_SPECTRA_CAPTION = (
    "Spectra of what was subtracted, not of residual EEG. Solid = beat-variable "
    "removal; dashed = heartbeat-locked removal. The 8-13 Hz band is both "
    "conventional alpha and BCG harmonics 8-13 at typical heart rates; it does "
    "not identify neural loss. Frequency minor ticks and the faint comb are 1 Hz "
    "(heart rate is not stored on the profile; epoch Welch resolution is ~1 Hz, "
    "so harmonics are not resolved as discrete lines). Phase-locking is locked / "
    "total removed power: a diagnostic of estimator class, not artifact purity."
)
_RATIOS_CAPTION = (
    "A uses chronological equal-beat blocks, not validation folds; more blocks "
    "raise the finite-sample noise floor. The 2-block ratio is the most optimistic "
    "local residual. Solid = as written; dashed = ECG-regressed (cannot see "
    "cardiac-shaped residual left in the file). B and C: one point per recording, "
    "summarised over participants. C is 8-13 Hz power in beat-variable removal "
    "relative to input; variable BCG and heartbeat-evoked neural activity are "
    "both possible."
)
_BCGNET_NOTE = (
    "BCGNet outputs include time used to train the subject model (70/15/15 split, "
    "then applied to the full recording). AAS and PCA-OBS are not trained that "
    "way. Residual panels are not a matched in-sample comparison."
)


def _caption_text(reference, groups, coverage, body):
    """The page caption, plus the provenance that qualifies how to read it."""
    parts = [_SHARED, body]
    support = reference[0].minimum_beats_per_block
    parts.append(f"Blocks below {support} beats are left missing, not replaced.")
    if any(
        profile.method == "bcgnet" for items in groups.values() for profile in items
    ):
        parts.append(_BCGNET_NOTE)
    if coverage:
        excluded = [
            f"{label} {count}"
            for label, count in coverage.items()
            if count and label in groups
        ]
        if excluded:
            parts.append(
                "Recordings without a usable output are excluded rather than "
                f"substituted ({'; '.join(excluded)})."
            )
    return " ".join(parts)


def save_profile_report(groups, *, title, output, coverage=None):
    groups = {label: items for label, items in groups.items() if items}
    if not groups:
        return False
    _validate_groups(groups)
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    reference = next(iter(groups.values()))
    counts = reference[0].block_counts
    finest = int(counts[0])
    participants = len({p.subject for p in reference})
    summarised = (
        "; bar = median and IQR over participants" if participants >= 3 else ""
    )
    discrete = dict(
        markersize=4.2, markeredgecolor="white", markeredgewidth=0.35
    )
    x_positions = np.arange(len(counts))
    n_groups = len(groups)
    jitter = np.linspace(-0.08, 0.08, n_groups) if n_groups > 1 else [0.0]
    arm_entries = [("FASTR", UNCORRECTED, UNCORRECTED_MARKER)]
    columns = []
    residual_series = []
    spectral_series = []
    ratio_series = []

    grid, fastr_local, _, _ = _anchor_band(reference, "local_wave_before", 0)
    _, fastr_pooled, _, _ = _anchor_band(reference, "pooled_before")
    residual_series.append((UNCORRECTED, None, fastr_local, fastr_pooled, None, None))

    for k, (label, profiles) in enumerate(groups.items()):
        color = arm_color(profiles[0].method)
        marker = arm_marker(profiles[0].method)
        arm_entries.append((label, color, marker))
        _, local_after, after_q1, after_q3 = _anchor_band(
            profiles, "local_wave_after", 0
        )
        _, pooled_after, _, _ = _anchor_band(profiles, "pooled_after")
        residual_series.append(
            (color, marker, local_after, pooled_after, after_q1, after_q3)
        )

        ratios, ratio_q1, ratio_q3 = _band(profiles, "local_ratio")
        by_participant = participant_values(profiles, "local_ratio")
        columns.append(
            (
                label,
                np.array([p.local_ratio[0, 0] for p in profiles], dtype=float),
                by_participant[:, 0, 0],
                np.array(
                    [p.variable_removal_alpha_ratio for p in profiles], dtype=float
                ),
                participant_values(profiles, "variable_removal_alpha_ratio"),
                color,
                marker,
            )
        )
        ratio_series.append(
            (color, marker, x_positions + jitter[k], ratios, ratio_q1, ratio_q3)
        )

        variable, var_q1, var_q3 = _band(profiles, "psd_removed_variable")
        locked = _centre(profiles, "psd_removed_locked")
        phase, phase_q1, phase_q3 = _band(profiles, "phase_locking_spectrum")
        spectral_series.append(
            (color, variable, var_q1, var_q3, locked, phase, phase_q1, phase_q3)
        )

    pages = report_page_paths(output)
    caption_kw = dict(reference=reference, groups=groups, coverage=coverage)

    with plt.rc_context(STYLE):
        fig, a = plt.subplots(figsize=RESIDUAL_SIZE, layout="constrained")
        fig.get_layout_engine().set(w_pad=0.02, h_pad=0.04)
        a.plot(
            grid, residual_series[0][2], color=UNCORRECTED, lw=1.35,
            label="FASTR", zorder=2.6, solid_capstyle="round",
        )
        a.plot(
            grid, residual_series[0][3], color=UNCORRECTED, lw=0.85, ls=DASH,
            alpha=0.9, zorder=2.2,
        )
        for (
            color, _marker, local_after, pooled_after, after_q1, after_q3
        ) in residual_series[1:]:
            fill_iqr(a, grid, after_q1, after_q3, color)
            a.plot(
                grid, local_after, color=color, lw=1.15, zorder=2.7,
                solid_capstyle="round",
            )
            a.plot(
                grid, pooled_after, color=color, lw=0.85, ls=DASH, alpha=0.9,
                zorder=2.3,
            )
        drawn = np.vstack([np.asarray(line.get_ydata()) for line in a.get_lines()])
        covered = np.flatnonzero(np.any(np.isfinite(drawn), axis=0))
        lo, hi = grid[0], grid[-1]
        if covered.size:
            # Cosine tapers at the correction-window edge pile up at the
            # aligned start; they are not the BCG peak.
            edge = 20.0
            lo = grid[covered[0]] + edge
            hi = grid[covered[-1]]
            a.set_xlim(lo, hi + 0.02 * (hi - lo))
        visible = []
        for line in a.get_lines():
            x = np.asarray(line.get_xdata(), dtype=float)
            y = np.asarray(line.get_ydata(), dtype=float)
            keep = np.isfinite(y) & (x >= lo) & (x <= hi)
            if np.any(keep):
                visible.append(y[keep])
        ymax = float(np.max(np.concatenate(visible))) if visible else 1.0
        a.set(
            xlabel="Time from BCG anchor (ms)",
            ylabel=r"RMS amplitude ($\mu\mathrm{V}$)",
            ylim=(0, ymax * 1.06),
        )
        a.axvline(0.0, color=FAINT, ls=DASH, lw=0.6, zorder=1)
        a.annotate(
            "BCG anchor", xy=(0, 0.97), xycoords=("data", "axes fraction"),
            ha="center", va="top", fontsize=5.8, color=MUTED,
        )
        delays = np.unique(
            np.round([p.applied_delay_seconds for p in reference], 6)
        )
        if delays.size == 1 and a.get_xlim()[0] <= -delays[0] * 1000 <= a.get_xlim()[1]:
            rpeak = -delays[0] * 1000.0
            a.axvline(rpeak, color=FAINT, ls=":", lw=0.6, zorder=1)
            a.annotate(
                "R-peak", xy=(rpeak, 0.97), xycoords=("data", "axes fraction"),
                ha="center", va="top", fontsize=5.8, color=MUTED,
            )
        panel(a, "A", "Heartbeat residual", "local blocks vs pooled over all beats")
        linestyle_key(a, [("-", f"local ({finest} blocks)"), (DASH, "pooled")])
        arm_legend(fig, arm_entries)
        fig.suptitle(
            title, fontsize=9.0, fontweight="bold", x=0.0, ha="left", color=INK
        )
        figure_caption(fig, _caption_text(**caption_kw, body=_RESIDUAL_CAPTION))
        save_figure(fig, pages["residual"], vector=True)
        plt.close(fig)

        fig, axes = plt.subplots(
            2, 1, figsize=SPECTRA_SIZE, sharex=True, layout="constrained"
        )
        fig.get_layout_engine().set(w_pad=0.02, h_pad=0.05, hspace=0.08)
        d, e = axes
        for (
            color, variable, var_q1, var_q3, locked, phase, phase_q1, phase_q3
        ) in spectral_series:
            fill_iqr(
                d, FREQUENCY_GRID_HZ,
                None if var_q1 is None else np.clip(var_q1, 1e-12, None),
                None if var_q3 is None else np.clip(var_q3, 1e-12, None),
                color,
            )
            d.plot(
                FREQUENCY_GRID_HZ, np.where(variable > 0, variable, np.nan),
                color=color, lw=1.25, zorder=2.7, solid_capstyle="round",
            )
            d.plot(
                FREQUENCY_GRID_HZ, np.where(locked > 0, locked, np.nan),
                color=color, lw=1.05, ls=DASH, zorder=2.5,
            )
            fill_iqr(e, FREQUENCY_GRID_HZ, phase_q1, phase_q3, color, alpha=0.10)
            e.plot(
                FREQUENCY_GRID_HZ, phase, color=color, lw=1.25, zorder=2.7,
                solid_capstyle="round",
            )
        has_positive_psd = any(
            np.any(np.asarray(line.get_ydata(), dtype=float) > 0)
            for line in d.get_lines()
        )
        d.set_ylabel(r"PSD ($\mu\mathrm{V}^2/\mathrm{Hz}$)")
        if has_positive_psd:
            d.set_yscale("log")
            d.set_ylim(*_positive_log_limits(d))
        else:
            d.set_ylim(0, 1.0)
        frequency_axis(d, max_hz=REPORT_MAX_HZ)
        panel(
            d, "A", "Removal spectra",
            "solid = variable; dashed = locked, not neural loss",
        )
        linestyle_key(d, [("-", "variable"), (DASH, "locked")])
        e.set_ylabel("Locked / total removed power")
        e.set(ylim=(0.0, 1.05), yticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        frequency_axis(e, max_hz=REPORT_MAX_HZ)
        panel(
            e, "B", "Removal phase-locking specificity",
            "diagnostic of estimator class, not artifact purity",
        )
        arm_legend(fig, arm_entries[1:])
        fig.suptitle(
            title, fontsize=9.0, fontweight="bold", x=0.0, ha="left", color=INK
        )
        figure_caption(fig, _caption_text(**caption_kw, body=_SPECTRA_CAPTION))
        save_figure(fig, pages["spectra"], vector=True)
        plt.close(fig)

        fig = plt.figure(figsize=RATIOS_SIZE, layout="constrained")
        fig.get_layout_engine().set(w_pad=0.02, h_pad=0.05, wspace=0.08, hspace=0.10)
        gs = fig.add_gridspec(2, 2, height_ratios=[1.08, 1.0])
        b = fig.add_subplot(gs[0, :])
        c = fig.add_subplot(gs[1, 0])
        f = fig.add_subplot(gs[1, 1])
        for color, marker, x_arm, ratios, ratio_q1, ratio_q3 in ratio_series:
            fill_iqr(
                b, x_arm,
                None if ratio_q1 is None else ratio_q1[0],
                None if ratio_q3 is None else ratio_q3[0],
                color,
            )
            b.plot(
                x_arm, ratios[0], color=color, lw=1.15, marker=marker,
                zorder=3, **discrete,
            )
            b.plot(
                x_arm, ratios[1], color=color, lw=0.9, ls=DASH, marker=marker,
                alpha=0.85, markersize=3.6, markerfacecolor="white",
                markeredgewidth=0.7, markeredgecolor=color, zorder=2.6,
            )
        b.axhline(1.0, color=FAINT, ls=":", lw=0.7, zorder=1)
        b.annotate(
            "no reduction", xy=(x_positions[-1] + 0.35, 1.0), xytext=(0, 2),
            textcoords="offset points", ha="right", fontsize=5.6, color=MUTED,
        )
        b.set(
            xlabel="Chronological blocks (evaluation resolution)",
            ylabel="After / before RMS ratio",
            xticks=x_positions,
            xticklabels=[str(count) for count in counts],
            xlim=(x_positions[0] - 0.4, x_positions[-1] + 0.4),
            ylim=(0, 1.12),
        )
        b.grid(axis="x", visible=False)
        panel(
            b, "A", "Residual across time scales",
            "IQR over participants" if participants >= 3 else "",
        )
        linestyle_key(
            b, [("-", "as written"), (DASH, "ECG regressed")], loc="lower right"
        )
        strip(
            c,
            [
                (label, points, summary, color, marker)
                for label, points, summary, _a, _as, color, marker in columns
            ],
            unit="After / before RMS ratio",
        )
        c.axhline(1.0, color=FAINT, ls=":", lw=0.7, zorder=1)
        c.set_ylim(bottom=0.0)
        panel(
            c, "B", "Residual ratio by recording",
            f"{finest} blocks, as written{summarised}",
        )
        strip(
            f,
            [
                (label, alpha_points, alpha_summary, color, marker)
                for label, _p, _s, alpha_points, alpha_summary, color, marker in columns
            ],
            unit="Variable 8-13 Hz / input (%)",
            percent=True,
        )
        f.set_ylim(bottom=0.0)
        panel(
            f, "C", "Beat-variable 8-13 Hz by recording",
            f"not identified as alpha{summarised}",
        )
        arm_legend(fig, arm_entries[1:])
        fig.suptitle(
            title, fontsize=9.0, fontweight="bold", x=0.0, ha="left", color=INK
        )
        figure_caption(fig, _caption_text(**caption_kw, body=_RATIOS_CAPTION))
        save_figure(fig, pages["ratios"], vector=True)
        plt.close(fig)
    written = set(pages.values())
    written.update(path.with_suffix(".pdf") for path in pages.values())
    for leftover in (output, output.with_suffix(".pdf")):
        if leftover.exists() and leftover not in written:
            leftover.unlink()
    return True


def save_correction_report(profile, *, title, output):
    return save_profile_report(
        {arm_label(profile.method): [profile]}, title=title, output=output
    )


def save_aggregate_report(profiles, *, title, output):
    if not profiles:
        return False
    return save_profile_report(
        {arm_label(profiles[0].method): profiles}, title=title, output=output
    )


def save_topography_report(groups, *, title, output):
    groups = {
        label: items
        for label, items in groups.items()
        if items and items[0].channel_names.size
    }
    if not groups:
        return False
    _validate_groups(groups)
    import matplotlib.pyplot as plt
    import mne

    reference = next(iter(groups.values()))
    names = reference[0].channel_names.tolist()
    info = mne.create_info(names, sfreq=1, ch_types="eeg")
    info.set_montage("standard_1020", on_missing="ignore")
    keep = [
        i
        for i, ch in enumerate(info["chs"])
        if np.all(np.isfinite(ch["loc"][:3])) and np.any(ch["loc"][:3])
    ]
    if len(keep) < 8:
        return False
    info = mne.pick_info(info, keep)
    counts = reference[0].block_counts

    def _limit(maps):
        finite = np.concatenate([np.asarray(m).ravel() for m in maps])
        finite = finite[np.isfinite(finite)]
        return max(float(finite.max()), 1e-12) if finite.size else 1.0

    # The uncorrected column is several times the corrected ones, so a single
    # shared scale flattens every corrected map into the bottom of the ramp and
    # the arms stop being comparable to each other. The uncorrected column keeps
    # its own scale; the arms share one sized to them. Each colourbar is
    # labelled, so no map is read against the wrong range.
    before = {row: _centre(reference, "topo_before")[row, keep] for row in
              range(len(counts))}
    after = {
        row: [_centre(items, "topo_after")[row, keep] for items in groups.values()]
        for row in range(len(counts))
    }
    before_max = _limit(list(before.values()))
    after_max = _limit([m for row in after.values() for m in row])

    def _draw(axis, values, vmax, cmap):
        if not np.all(np.isfinite(values)):
            axis.text(
                0.5,
                0.5,
                "insufficient\nbeats",
                ha="center",
                va="center",
                transform=axis.transAxes,
                fontsize=6.0,
                color=MUTED,
            )
            axis.axis("off")
            return None
        image, contours = mne.viz.plot_topomap(
            values,
            info,
            axes=axis,
            show=False,
            cmap=cmap,
            vlim=(0, vmax),
            contours=5,
            sensors=True,
            res=128,
            extrapolate="head",
            outlines="head",
        )
        image.set_rasterized(True)
        if contours is not None and hasattr(contours, "set_linewidths"):
            contours.set_linewidths(0.35)
        return image

    n_cols = len(groups) + 1
    n_rows = len(counts) + 1
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(7.24, min(9.6, TOPOGRAPHY_ROW_IN * n_rows + 1.15)),
            squeeze=False,
            layout="constrained",
        )
        fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.04, hspace=0.06)
        before_image = None
        after_image = None
        for row in range(len(counts)):
            maps = [before[row], *after[row]]
            for col, (axis, values, label) in enumerate(
                zip(axes[row], maps, ["FASTR", *groups], strict=True)
            ):
                if row == 0:
                    axis.set_title(
                        label, fontsize=7.5, fontweight="bold", pad=6, color=INK
                    )
                if col == 0:
                    axis.text(
                        -0.12,
                        0.5,
                        f"{counts[row]} blocks",
                        transform=axis.transAxes,
                        ha="right",
                        va="center",
                        fontsize=7.0,
                        fontweight="bold",
                        color=INK,
                    )
                image = _draw(
                    axis,
                    values,
                    before_max if col == 0 else after_max,
                    RMS_CMAP,
                )
                if image is None:
                    continue
                if col == 0:
                    before_image = image
                else:
                    after_image = image

        # Bottom row: beat-variable alpha. The empty first cell is left blank
        # rather than filled with a caption card; the reading guide is one
        # caption at the foot of the page.
        axes[-1, 0].axis("off")
        axes[-1, 0].text(
            0.98,
            0.5,
            "beat-variable\n8-13 Hz",
            transform=axes[-1, 0].transAxes,
            ha="right",
            va="center",
            fontsize=7.0,
            fontweight="bold",
            color=INK,
        )
        alpha_maps = [
            _centre(items, "topo_variable_alpha_ratio")[keep] * 100
            for items in groups.values()
        ]
        alpha_max = _limit(alpha_maps)
        alpha_image = None
        for axis, values, label in zip(axes[-1, 1:], alpha_maps, groups, strict=True):
            axis.set_title(label, fontsize=7.0, fontweight="bold", pad=5, color=INK)
            image = _draw(axis, values, alpha_max, ALPHA_CMAP)
            if image is not None:
                alpha_image = image

        def _bar(image, axis_list, label):
            """A horizontal bar under its own columns.

            Vertical bars attached to the uncorrected column are laid out
            between it and the first arm, which cuts the grid in half.
            """
            if image is None:
                return
            # Shrink against the span so a bar under three columns does not
            # come out three times the length of the one under a single column.
            bar = fig.colorbar(
                image,
                ax=axis_list,
                location="bottom",
                shrink=min(0.82, 1.55 / max(len(axis_list) // max(len(counts), 1), 1)),
                aspect=32,
                pad=0.02,
            )
            bar.set_label(label, fontsize=6.2, color=INK, labelpad=1.5)
            bar.ax.tick_params(labelsize=5.6, length=2.0, width=0.45, pad=1.2)
            bar.outline.set_linewidth(0.4)
            bar.outline.set_edgecolor(FAINT)

        block_rows = list(range(len(counts)))
        _bar(
            before_image,
            [axes[row, 0] for row in block_rows],
            r"Uncorrected local RMS ($\mu\mathrm{V}$)",
        )
        _bar(
            after_image,
            [axes[row, col] for row in block_rows for col in range(1, len(groups) + 1)],
            r"Corrected local RMS ($\mu\mathrm{V}$)",
        )
        _bar(
            alpha_image,
            list(axes[-1, 1:]),
            "Variable 8-13 Hz removal / input (%)",
        )

        fig.suptitle(
            title, fontsize=9.0, fontweight="bold", x=0.0, ha="left", color=INK
        )
        figure_caption(
            fig,
            "Local heartbeat RMS before and after correction. The uncorrected "
            "column and the corrected arms carry separate scales, each with its "
            "own colourbar: on one shared scale the corrected maps collapse into "
            "the bottom of the ramp and cannot be compared with each other. "
            "Absolute magnitudes are therefore comparable within a colourbar, not "
            "across them. All maps use saved EEG and participant-balanced "
            "summaries. More blocks raise the noise floor. Spatial similarity does "
            "not establish whether removed activity was neural.",
        )
        save_figure(fig, output, vector=True)
        plt.close(fig)
    return True
