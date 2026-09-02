"""Correction reports at three levels, all drawn with the same six panels.

A recording, a subject, and the cohort get the *same* figure: heartbeat-locked
waveforms and spectra. Only the averaging changes. Reading one teaches you to
read all three, and a subject page can be held against the cohort page by eye
because every axis means the same thing.

The panels answer, in order:

1. did the heartbeat-locked artifact go away (A, C),
2. does what was removed have the shape and spectrum of BCG (B, D),
3. how much of what was removed was *not* cardiac-locked, and therefore cannot
   have been BCG (D, F).

Panel D is the one that separates a correction from an aggressive filter. A
method that subtracts more than the artifact lowers every residual ratio while
taking neural signal with it, so suppression alone cannot be read as quality --
the non-phase-locked trace has to stay low as well.

Each recording writes a small ``*_profile.npz`` holding exactly the traces the
panels need, so subject and cohort pages are built by averaging those instead of
re-reading gigabytes of EEG.

Waveform panels use a fixed grid relative to the **R peak**, which is what makes
them averageable across recordings whose estimated ECG-to-BCG delays differ. The
scalar metrics keep the correction's own delay-centred window, so
``locked_ratio`` on the page is the same number as ``residual_qc.ratio`` in the
provenance written beside it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, cast

import matplotlib
import numpy as np
import numpy.typing as npt
from scipy.signal import welch

from .figure_style import (
    ALPHA_BAND_HZ,
    ARTIFACT,
    COLLATERAL,
    CORRECTED,
    FAINT,
    INK,
    MUTED,
    STYLE,
    UNCORRECTED,
    figure_caption,
    full_ylim,
    harmonics,
    legend,
    log_ylim,
    panel,
    quantile_band,
    save_figure,
    shade_alpha,
    spectrum_summary,
    zero_lines,
)
from .metrics import (
    delay_estimation_eeg,
    is_posterior_eeg_channel,
    regress_out_reference,
    trigger_locked_template,
)

REPORT_MAX_HZ = 45.0
EXCERPT_SECONDS = 4.0
#: Shared grids. Everything is interpolated onto these so profiles from
#: different recordings average without re-reading any EEG.
DISPLAY_MS = np.arange(-200.0, 901.0)
FREQUENCY_GRID_HZ = np.arange(1.0, REPORT_MAX_HZ + 0.001, 0.25)
MINIMUM_REPORT_BEATS = 8
PROFILE_SCHEMA_VERSION = 1

_TEXT_FIELDS = frozenset({"method", "label"})
_SCALAR_FIELDS = frozenset({
    "locked_ratio", "locked_before_uv", "locked_after_uv", "specificity",
    "alpha_collateral_fraction", "beats", "heart_rate_bpm",
    "applied_delay_seconds", "gap_fraction", "locked_ratio_raw",
})
_WAVE_FIELDS = ("template_before", "template_after", "removed_locked")
_SPECTRUM_FIELDS = (
    "psd_before", "psd_after", "psd_present",
    "psd_removed_locked", "psd_removed_nonlocked",
)
_TRACE_FIELDS = _WAVE_FIELDS + _SPECTRUM_FIELDS



@dataclass(frozen=True, slots=True)
class CorrectionProfile:
    """Everything the six panels draw, on grids shared across recordings."""

    method: str
    label: str
    template_before: np.ndarray
    template_after: np.ndarray
    removed_locked: np.ndarray
    psd_before: np.ndarray
    psd_after: np.ndarray
    psd_present: np.ndarray
    psd_removed_locked: np.ndarray
    psd_removed_nonlocked: np.ndarray
    excerpt_seconds: np.ndarray
    excerpt_before: np.ndarray
    excerpt_after: np.ndarray
    # Per-channel maps, over every EEG channel rather than the posterior subset
    # the scalar metrics use. EEG is spatial: where a method takes its
    # collateral is what says whether the collateral was neural.
    channel_names: np.ndarray
    topo_artifact: np.ndarray
    topo_alpha_present: np.ndarray
    topo_removed_locked: np.ndarray
    topo_collateral_alpha: np.ndarray
    locked_ratio: float
    locked_before_uv: float
    locked_after_uv: float
    specificity: float
    alpha_collateral_fraction: float
    beats: int
    heart_rate_bpm: float
    applied_delay_seconds: float
    gap_fraction: float
    #: The locked ratio measured on the EEG as written, with no ECG regression.
    #: Every other scalar regresses the ECG channel out first, which is right
    #: for separating BCG from volume-conducted QRS but blind to ECG-shaped
    #: residual a method leaves in the file. A gap between the two ratios is
    #: exactly that residual.
    locked_ratio_raw: float = float("nan")


def compute_correction_profile(
    before_volts: npt.ArrayLike,
    after_volts: npt.ArrayLike,
    channel_names: tuple[str, ...],
    *,
    ecg_channel_index: int,
    peak_samples: npt.ArrayLike,
    sampling_rate_hz: float,
    delay_seconds: float,
    window_seconds: tuple[float, float],
    gap_fraction: float,
    method: str,
    label: str = "",
) -> CorrectionProfile | None:
    """Measure one correction. ``None`` when there are too few complete beats."""
    before = np.asarray(before_volts, dtype=np.float64)
    after = np.asarray(after_volts, dtype=np.float64)
    peaks = np.asarray(peak_samples, dtype=np.int64)
    n_samples = before.shape[1]

    posterior_before = delay_estimation_eeg(
        before, channel_names, ecg_channel_index=ecg_channel_index
    ) * 1e6
    posterior_after = delay_estimation_eeg(
        after, channel_names, ecg_channel_index=ecg_channel_index
    ) * 1e6
    # Match cardiac_locked_rms: strip each channel's recording-level median, or
    # an amplifier offset is counted as cardiac-locked energy and the ratio
    # collapses toward 1.
    posterior_before -= np.median(posterior_before, axis=1, keepdims=True)
    posterior_after -= np.median(posterior_after, axis=1, keepdims=True)
    removed = posterior_before - posterior_after

    metric = _epoch_starts(
        peaks, delay_seconds + window_seconds[0], delay_seconds + window_seconds[1],
        sampling_rate_hz, n_samples,
    )
    display = _epoch_starts(
        peaks, DISPLAY_MS[0] / 1e3, (DISPLAY_MS[-1] + 1.0) / 1e3,
        sampling_rate_hz, n_samples,
    )
    if metric is None or display is None:
        return None
    metric_starts, metric_span = metric
    display_starts, display_span = display

    template_before = trigger_locked_template(
        posterior_before, metric_starts.astype(np.float64), epoch_samples=metric_span
    )
    template_after = trigger_locked_template(
        posterior_after, metric_starts.astype(np.float64), epoch_samples=metric_span
    )
    locked_before_uv = float(np.median(_rms(template_before)))
    locked_after_uv = float(np.median(_rms(template_after)))
    locked_ratio = (
        locked_after_uv / locked_before_uv if locked_before_uv else float("nan")
    )
    locked_ratio_raw = _locked_ratio_as_written(
        before, after, channel_names, ecg_channel_index, metric_starts, metric_span
    )

    index = display_starts[:, None] + np.arange(display_span)[None, :]
    removed_epochs = removed[:, index]
    removed_locked = removed_epochs.mean(axis=1)
    removed_nonlocked = removed_epochs - removed_locked[:, None, :]
    total_rms = np.sqrt(np.mean(removed_epochs**2, axis=(1, 2)))
    locked_rms = _rms(removed_locked)
    with np.errstate(invalid="ignore", divide="ignore"):
        specificity = float(
            np.median(locked_rms / np.where(total_rms > 0, total_rms, np.nan))
        )

    channels = removed.shape[0]
    nperseg = int(min(display_span, 1024))
    present = posterior_before[:, index].reshape(channels, -1)
    collateral = removed_nonlocked.reshape(channels, -1)
    raw_alpha = _band_power(present, sampling_rate_hz, nperseg, ALPHA_BAND_HZ)
    alpha_fraction = (
        _band_power(collateral, sampling_rate_hz, nperseg, ALPHA_BAND_HZ) / raw_alpha
        if raw_alpha else float("nan")
    )

    # Every EEG channel, ECG regressed out, for the spatial maps. The scalar
    # metrics deliberately use posterior channels only; topography must not.
    eeg_picks = np.asarray(
        [i for i in range(before.shape[0]) if i != ecg_channel_index], dtype=np.int64
    )
    ecg = ecg_channel_index
    all_before = regress_out_reference(before[eeg_picks], before[ecg]) * 1e6
    all_after = regress_out_reference(after[eeg_picks], after[ecg]) * 1e6
    all_before -= np.median(all_before, axis=1, keepdims=True)
    all_after -= np.median(all_after, axis=1, keepdims=True)
    all_removed = (all_before - all_after)[:, index]
    all_locked = all_removed.mean(axis=1)
    all_nonlocked = (all_removed - all_locked[:, None, :]).reshape(len(eeg_picks), -1)
    topo_artifact = _rms(all_before[:, index].mean(axis=1))
    topo_removed_locked = _rms(all_locked)
    topo_alpha_present = _channel_band_power(
        all_before[:, index].reshape(len(eeg_picks), -1),
        sampling_rate_hz, nperseg, ALPHA_BAND_HZ,
    )
    topo_collateral_alpha = _channel_band_power(
        all_nonlocked, sampling_rate_hz, nperseg, ALPHA_BAND_HZ
    )

    intervals = np.diff(peaks) / sampling_rate_hz
    heart_rate = 60.0 / float(np.median(intervals)) if intervals.size else float("nan")
    display_ms = (
        np.arange(display_span) + round(DISPLAY_MS[0] / 1e3 * sampling_rate_hz)
    ) / sampling_rate_hz * 1e3
    long_nperseg = int(min(20 * sampling_rate_hz, posterior_before.shape[1]))

    return CorrectionProfile(
        method=method,
        label=label,
        template_before=_to_grid(
            display_ms, _channel_mean_template(posterior_before, index)
        ),
        template_after=_to_grid(
            display_ms, _channel_mean_template(posterior_after, index)
        ),
        removed_locked=_to_grid(display_ms, removed_locked.mean(axis=0)),
        psd_before=_spectrum(
            posterior_before.mean(axis=0), sampling_rate_hz, long_nperseg
        ),
        psd_after=_spectrum(
            posterior_after.mean(axis=0), sampling_rate_hz, long_nperseg
        ),
        psd_present=_spectrum(present, sampling_rate_hz, nperseg),
        psd_removed_locked=_spectrum(removed_locked, sampling_rate_hz, nperseg),
        psd_removed_nonlocked=_spectrum(collateral, sampling_rate_hz, nperseg),
        **_excerpt(posterior_before, posterior_after, peaks, sampling_rate_hz),
        channel_names=np.asarray(
            [channel_names[int(i)] for i in eeg_picks], dtype="<U16"
        ),
        topo_artifact=topo_artifact,
        topo_alpha_present=topo_alpha_present,
        topo_removed_locked=topo_removed_locked,
        topo_collateral_alpha=topo_collateral_alpha,
        locked_ratio=locked_ratio,
        locked_before_uv=locked_before_uv,
        locked_after_uv=locked_after_uv,
        specificity=specificity,
        alpha_collateral_fraction=alpha_fraction,
        beats=int(metric_starts.size),
        heart_rate_bpm=heart_rate,
        applied_delay_seconds=float(delay_seconds),
        gap_fraction=float(gap_fraction),
        locked_ratio_raw=locked_ratio_raw,
    )


def _locked_ratio_as_written(
    before, after, channel_names, ecg_channel_index, starts, span
) -> float:
    """The same locked ratio, on the posterior EEG exactly as it was written."""
    picks = np.asarray([
        index for index, name in enumerate(channel_names)
        if index != ecg_channel_index and is_posterior_eeg_channel(name)
    ], dtype=np.int64)
    if picks.size == 0:
        return float("nan")
    values = []
    for data in (before, after):
        posterior = data[picks] * 1e6
        posterior = posterior - np.median(posterior, axis=1, keepdims=True)
        template = trigger_locked_template(
            posterior, starts.astype(np.float64), epoch_samples=span
        )
        values.append(float(np.median(_rms(template))))
    return values[1] / values[0] if values[0] else float("nan")


def write_profile(profile: CorrectionProfile, path: Path) -> None:
    """Persist a profile so subject and cohort pages need no raw EEG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        schema_version=np.asarray(PROFILE_SCHEMA_VERSION),
        **{key: np.asarray(value) for key, value in asdict(profile).items()},
    )


def read_profile(path: Path) -> CorrectionProfile:
    with np.load(path, allow_pickle=False) as data:
        expected = {field.name for field in fields(CorrectionProfile)}
        expected.add("schema_version")
        actual = set(data.files)
        if actual != expected:
            missing = sorted(expected - actual)
            unexpected = sorted(actual - expected)
            details = []
            if missing:
                details.append(f"missing {', '.join(missing)}")
            if unexpected:
                details.append(f"unexpected {', '.join(unexpected)}")
            raise ValueError(f"invalid profile schema ({'; '.join(details)}): {path}")
        version = int(data["schema_version"])
        if version != PROFILE_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported profile schema_version {version}; "
                f"expected {PROFILE_SCHEMA_VERSION}: {path}"
            )
        values: dict[str, object] = {}
        for field in fields(CorrectionProfile):
            value = data[field.name]
            if field.name in _TEXT_FIELDS:
                values[field.name] = str(value)
            elif field.name == "beats":
                values[field.name] = int(value)
            elif field.name in _SCALAR_FIELDS:
                values[field.name] = float(value)
            else:
                values[field.name] = np.asarray(value)
    return CorrectionProfile(**cast(dict[str, Any], values))


def save_correction_report(
    profile: CorrectionProfile, *, title: str, output: Path
) -> None:
    """The single-recording page."""
    _render([profile], title=title, output=output, aggregate=False)


def save_aggregate_report(
    profiles: list[CorrectionProfile], *, title: str, output: Path
) -> bool:
    """The same page for a subject or the cohort: grand average of profiles."""
    if not profiles:
        return False
    _render(profiles, title=title, output=output, aggregate=True)
    return True


_CAPTION = (
    "Posterior EEG channels; the ECG channel is regressed out before every "
    "measurement except the ratio marked 'as written'. Time is relative to the ECG "
    "R peak (dotted line). Shaded band: alpha, 8-13 Hz. Ticks above a spectrum "
    "mark cardiac harmonics. Specificity is the share of the removed amplitude "
    "that is heartbeat-locked; alpha collateral is the share of the recording's "
    "alpha power removed in the non-locked part. Both sit near their best value "
    "for any beat-invariant template whatever its shape, so they say how "
    "adaptive a method is, not that it is right; read them beside the locked "
    "ratio. Neither sees ECG-shaped residual: a gap between the two ratios in F "
    "is cardiac field left in the file."
)
_CAPTION_AGGREGATE = (
    "Lines are the mean across recordings (waveforms) or the median (spectra); "
    "bands are the interquartile range, and in E also the 5-95% range. " + _CAPTION
)


def _render(
    profiles: list[CorrectionProfile], *, title: str, output: Path, aggregate: bool
) -> None:
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    stack = {
        name: np.vstack([getattr(profile, name) for profile in profiles])
        for name in _TRACE_FIELDS
    }
    centre = {name: np.nanmean(stack[name], axis=0) for name in _WAVE_FIELDS}
    centre.update({name: spectrum_summary(stack[name]) for name in _SPECTRUM_FIELDS})

    with plt.rc_context(STYLE):
        figure, axes = plt.subplots(2, 3, figsize=(14.0, 7.8), layout="constrained")
        _panel_locked(axes[0, 0], stack, centre, profiles, aggregate)
        _panel_removed_shape(axes[0, 1], centre)
        _panel_spectra(axes[0, 2], centre, profiles)
        _panel_removed_spectra(axes[1, 0], centre, profiles)
        if aggregate:
            _panel_residual_spread(axes[1, 1], stack, profiles)
        else:
            _panel_excerpt(axes[1, 1], profiles[0])
        _panel_summary(axes[1, 2], profiles, aggregate)
        figure.suptitle(title, fontsize=11.5, x=0.01, ha="left", color=INK)
        figure_caption(figure, _CAPTION_AGGREGATE if aggregate else _CAPTION)
        save_figure(figure, output, vector=aggregate)
        plt.close(figure)


def _rms(values: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(values**2, axis=-1))


def _epoch_starts(peaks, start_seconds, stop_seconds, sampling_rate, n_samples):
    start = round(start_seconds * sampling_rate)
    span = round(stop_seconds * sampling_rate) - start
    starts = np.asarray(peaks, dtype=np.int64) + start
    keep = (starts >= 0) & (starts + span <= n_samples)
    if span < 2 or int(np.count_nonzero(keep)) < MINIMUM_REPORT_BEATS:
        return None
    return starts[keep], span


def _channel_mean_template(data: np.ndarray, index: np.ndarray) -> np.ndarray:
    return data[:, index].mean(axis=1).mean(axis=0)


def _to_grid(times_ms: np.ndarray, values: np.ndarray) -> np.ndarray:
    return np.interp(DISPLAY_MS, times_ms, values, left=np.nan, right=np.nan)


def _spectrum(data: np.ndarray, sampling_rate: float, nperseg: int) -> np.ndarray:
    freqs, power = welch(data, fs=sampling_rate, nperseg=nperseg, axis=-1)
    if power.ndim > 1:
        power = np.median(power, axis=0)
    return np.interp(FREQUENCY_GRID_HZ, freqs, power, left=np.nan, right=np.nan)


def _channel_band_power(data, sampling_rate, nperseg, band) -> np.ndarray:
    """Band power per channel, for a topography rather than a single number."""
    freqs, power = welch(data, fs=sampling_rate, nperseg=nperseg, axis=-1)
    inside = (freqs >= band[0]) & (freqs <= band[1])
    return power[:, inside].sum(axis=1)


def _band_power(data, sampling_rate, nperseg, band) -> float:
    freqs, power = welch(data, fs=sampling_rate, nperseg=nperseg, axis=-1)
    inside = (freqs >= band[0]) & (freqs <= band[1])
    return float(np.median(power[:, inside].sum(axis=1)))


def _excerpt(before, after, peaks, sampling_rate) -> dict[str, np.ndarray]:
    span = int(EXCERPT_SECONDS * sampling_rate)
    start = int(min(max(int(peaks[0]) - int(sampling_rate), 0),
                    max(before.shape[1] - span, 0)))
    stop = start + span
    channel = int(np.argmax(_rms(before[:, start:stop])))
    excerpt_before = before[channel, start:stop]
    excerpt_after = after[channel, start:stop]
    return {
        "excerpt_seconds": np.arange(stop - start) / sampling_rate,
        "excerpt_before": excerpt_before - excerpt_before.mean(),
        "excerpt_after": excerpt_after - excerpt_after.mean(),
    }


def _median_hr(profiles) -> float:
    return float(np.nanmedian([profile.heart_rate_bpm for profile in profiles]))


def _median_of(profiles, name: str) -> float:
    return float(np.nanmedian([getattr(profile, name) for profile in profiles]))


def _time_axis(axis) -> None:
    axis.set_xlabel("time from R peak (ms)")
    axis.set_ylabel("amplitude (µV)")
    axis.set_xlim(DISPLAY_MS[0], DISPLAY_MS[-1])


def _frequency_axis(axis) -> None:
    axis.set_xlabel("frequency (Hz)")
    axis.set_ylabel("PSD (µV²/Hz)")
    axis.set_xlim(0.0, REPORT_MAX_HZ)


def _panel_locked(axis, stack, centre, profiles, aggregate) -> None:
    if aggregate:
        quantile_band(axis, DISPLAY_MS, stack["template_before"], UNCORRECTED,
                      alpha=0.16)
        quantile_band(axis, DISPLAY_MS, stack["template_after"], CORRECTED,
                      alpha=0.22)
    axis.plot(DISPLAY_MS, centre["template_before"], color=UNCORRECTED, lw=1.8,
              label="before correction")
    axis.plot(DISPLAY_MS, centre["template_after"], color=CORRECTED, lw=1.8,
              label="after correction")
    zero_lines(axis)
    if aggregate:
        low, high = np.nanpercentile(stack["template_before"], [25.0, 75.0], axis=0)
        full_ylim(axis, low, high, centre["template_before"],
                  centre["template_after"])
    else:
        full_ylim(axis, centre["template_before"], centre["template_after"])
    ratio = _median_of(profiles, "locked_ratio")
    panel(
        axis, "A", "Heartbeat-locked average",
        f"{_median_of(profiles, 'locked_before_uv'):.1f} to "
        f"{_median_of(profiles, 'locked_after_uv'):.1f} µV,  "
        f"{'median ' if aggregate else ''}ratio {ratio:.3f}",
    )
    _time_axis(axis)
    legend(axis)


def _panel_removed_shape(axis, centre) -> None:
    axis.plot(DISPLAY_MS, centre["template_before"], color=UNCORRECTED, lw=1.3,
              ls="--", label="artifact, before correction")
    axis.plot(DISPLAY_MS, centre["removed_locked"], color=ARTIFACT, lw=1.8,
              label="removed, cardiac-locked")
    zero_lines(axis)
    full_ylim(axis, centre["template_before"], centre["removed_locked"])
    panel(axis, "B", "Cardiac-locked component removed")
    _time_axis(axis)
    legend(axis)


def _panel_spectra(axis, centre, profiles) -> None:
    axis.semilogy(FREQUENCY_GRID_HZ, centre["psd_before"], color=UNCORRECTED, lw=1.3,
                  label="before correction")
    axis.semilogy(FREQUENCY_GRID_HZ, centre["psd_after"], color=CORRECTED, lw=1.3,
                  label="after correction")
    harmonics(axis, _median_hr(profiles))
    shade_alpha(axis)
    log_ylim(axis, centre["psd_before"], centre["psd_after"], decades=3.0)
    panel(axis, "C", "Power spectrum, posterior mean")
    _frequency_axis(axis)
    legend(axis)


def _panel_removed_spectra(axis, centre, profiles) -> None:
    axis.semilogy(FREQUENCY_GRID_HZ, centre["psd_present"], color=MUTED, lw=1.2,
                  label="present in recording")
    axis.semilogy(FREQUENCY_GRID_HZ, centre["psd_removed_locked"], color=ARTIFACT,
                  lw=1.4, label="removed, cardiac-locked (artifact)")
    axis.semilogy(FREQUENCY_GRID_HZ, centre["psd_removed_nonlocked"],
                  color=COLLATERAL, lw=1.4,
                  label="removed, not cardiac-locked (collateral)")
    harmonics(axis, _median_hr(profiles))
    shade_alpha(axis)
    log_ylim(axis, centre["psd_present"], centre["psd_removed_locked"],
             centre["psd_removed_nonlocked"], decades=4.5)
    panel(axis, "D", "Removed power, split by cardiac locking")
    _frequency_axis(axis)
    legend(axis)


def _panel_excerpt(axis, profile) -> None:
    removed = profile.excerpt_before - profile.excerpt_after
    floor = float(np.min(profile.excerpt_before)) - float(np.ptp(removed)) * 0.55
    axis.plot(profile.excerpt_seconds, profile.excerpt_before, color=UNCORRECTED,
              lw=0.7, label="before")
    axis.plot(profile.excerpt_seconds, profile.excerpt_after, color=CORRECTED,
              lw=0.7, label="after")
    axis.plot(profile.excerpt_seconds, removed + floor, color=ARTIFACT, lw=0.7,
              label="removed (offset)")
    axis.axhline(floor, color=FAINT, lw=0.6)
    panel(axis, "E", f"{EXCERPT_SECONDS:.0f} s excerpt, largest-amplitude channel")
    axis.set_xlabel("time (s)")
    axis.set_ylabel("amplitude (µV)")
    axis.set_xlim(profile.excerpt_seconds[0], profile.excerpt_seconds[-1])
    legend(axis, loc="upper right", ncols=3)


def _panel_residual_spread(axis, stack, profiles) -> None:
    """What is *left* after correction, across recordings."""
    residual = stack["template_after"]
    quantile_band(axis, DISPLAY_MS, residual, CORRECTED, quantiles=(5.0, 95.0),
                  alpha=0.12, label="5-95% of recordings")
    quantile_band(axis, DISPLAY_MS, residual, CORRECTED, alpha=0.28,
                  label="interquartile range")
    axis.plot(DISPLAY_MS, np.nanmean(residual, axis=0), color=CORRECTED, lw=1.8,
              label="mean residual")
    zero_lines(axis)
    # Scale to the interquartile band: one recording's outlier at the R peak
    # must not flatten the residual every other recording shares.
    low, high = np.nanpercentile(residual, [25.0, 75.0], axis=0)
    full_ylim(axis, low, high, np.nanmean(residual, axis=0), pad=0.3)
    panel(axis, "E", "Residual after correction", f"{len(profiles)} recordings")
    _time_axis(axis)
    legend(axis)


def _panel_summary(axis, profiles, aggregate) -> None:
    axis.axis("off")
    panel(axis, "F", "Summary",
          "median [Q1 to Q3] across recordings" if aggregate else "")

    def stat(name: str, digits: int = 3, scale: float = 1.0, unit: str = "") -> str:
        values = np.asarray(
            [getattr(profile, name) for profile in profiles], dtype=float
        ) * scale
        values = values[np.isfinite(values)]
        if values.size == 0:
            return "not in profile; rerun bcg reports"
        if not aggregate:
            return f"{values[0]:.{digits}f}{unit}"
        q1, median, q3 = np.percentile(values, [25, 50, 75])
        return f"{median:.{digits}f}  [{q1:.{digits}f} to {q3:.{digits}f}]{unit}"

    rows = (
        ("method", profiles[0].method.replace("_", " ")),
        ("recordings", f"{len(profiles)}"),
        ("heartbeats", f"{int(np.sum([p.beats for p in profiles]))}"),
        ("heart rate", f"{_median_hr(profiles):.1f} bpm"),
        ("applied ECG-to-BCG delay",
         f"{_median_of(profiles, 'applied_delay_seconds') * 1e3:.0f} ms"),
        None,
        ("locked residual, before", stat("locked_before_uv", 2, unit=" µV")),
        ("locked residual, after", stat("locked_after_uv", 2, unit=" µV")),
        ("locked ratio, ECG regressed", stat("locked_ratio")),
        ("locked ratio, as written", stat("locked_ratio_raw")),
        None,
        ("specificity of removal", stat("specificity")),
        ("alpha taken as collateral",
         stat("alpha_collateral_fraction", 1, 100.0, " %")),
    )
    y = 0.97
    for row in rows:
        if row is None:
            y -= 0.035
            continue
        label, value = row
        axis.text(0.0, y, label, va="center", ha="left", color=MUTED, fontsize=8.6)
        axis.text(1.0, y, value, va="center", ha="right", color=INK, fontsize=8.6)
        y -= 0.068


def _montage_info(names: list[str]):
    import mne

    with mne.utils.use_log_level("ERROR"):
        probe = mne.create_info(names, sfreq=1.0, ch_types="eeg")
        probe.set_montage(
            mne.channels.make_standard_montage("standard_1020"), on_missing="ignore"
        )
        positioned = [
            name for name, channel in zip(names, probe["chs"], strict=True)
            if np.all(np.isfinite(channel["loc"][:3])) and np.any(channel["loc"][:3])
        ]
        if len(positioned) < 8:
            return None, None
        info = mne.create_info(positioned, sfreq=1.0, ch_types="eeg")
        info.set_montage(
            mne.channels.make_standard_montage("standard_1020"), on_missing="ignore"
        )
    return info, np.asarray([names.index(name) for name in positioned])


def save_topography_report(
    groups: dict[str, list[CorrectionProfile]],
    *,
    title: str,
    output: Path,
) -> bool:
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    import mne

    labels = [
        name for name, profiles in groups.items()
        if profiles and np.size(profiles[0].channel_names)
    ]
    if not labels:
        return False
    names = [str(n) for n in groups[labels[0]][0].channel_names]
    info, keep = _montage_info(names)
    if info is None:
        return False

    def mean_map(label: str, field: str) -> np.ndarray:
        stack = np.vstack([getattr(p, field) for p in groups[label]])
        return np.nanmean(stack, axis=0)[keep]

    reference = labels[0]
    artifact = mean_map(reference, "topo_artifact")
    removed = [mean_map(label, "topo_removed_locked") for label in labels]
    alpha = mean_map(reference, "topo_alpha_present")
    with np.errstate(invalid="ignore", divide="ignore"):
        shares = [
            np.where(alpha > 0, mean_map(label, "topo_collateral_alpha") / alpha, 0.0)
            * 100.0
            for label in labels
        ]
    columns = len(labels) + 1

    def draw(axis, values, vmax, label):
        with mne.utils.use_log_level("ERROR"):
            image, _ = mne.viz.plot_topomap(
                values, info, axes=axis, show=False, cmap="viridis",
                vlim=(0.0, vmax), contours=4, sensors=True,
            )
        axis.set_title(label, fontsize=9.5, color=INK, loc="center")
        return image

    def colorbar(image, axis, text):
        bar = figure.colorbar(image, ax=axis, fraction=0.05, pad=0.04, aspect=18)
        bar.set_label(text, fontsize=8)
        bar.ax.tick_params(labelsize=7)

    with plt.rc_context(STYLE):
        figure, axes = plt.subplots(
            2, columns, figsize=(3.0 * columns + 0.8, 6.6),
            layout="constrained", squeeze=False,
        )
        for axis in axes.ravel():
            axis.grid(False)
        scale_a = float(np.nanmax(np.abs(np.concatenate([artifact, *removed]))))
        image = draw(axes[0][0], artifact, scale_a, "artifact present")
        for column, (values, label) in enumerate(zip(removed, labels, strict=True), 1):
            image = draw(axes[0][column], values, scale_a, label)
        colorbar(image, axes[0][-1], "cardiac-locked amplitude (µV)")

        image = draw(axes[1][0], alpha, float(np.nanmax(alpha)), "alpha present")
        colorbar(image, axes[1][0], "alpha power (µV²)")
        scale_b = float(max(np.nanmax(np.concatenate(shares)), 1e-9))
        for column, (values, label) in enumerate(zip(shares, labels, strict=True), 1):
            image = draw(axes[1][column], values, scale_b, label)
        colorbar(image, axes[1][-1], "alpha removed as collateral (%)")

        for row, letter in enumerate("AB"):
            axes[row][0].annotate(
                letter, xy=(0.0, 1.0), xycoords="axes fraction",
                xytext=(-10.0, 8.0), textcoords="offset points",
                fontsize=12.0, fontweight="bold", color=INK,
            )
        figure.suptitle(title, fontsize=11.5, x=0.01, ha="left", color=INK)
        figure_caption(
            figure,
            "Row A: heartbeat-locked amplitude before correction, and the "
            "cardiac-locked amplitude each method removed, on one shared scale. "
            "Row B: alpha-band power before correction (own scale), and the "
            "share of that alpha each method removed in the part of its removal "
            "that was not heartbeat-locked, on one scale across methods. A "
            "collateral map organised like the alpha map is neural signal loss. "
            "All maps are ECG-regressed cohort means over every EEG channel.",
            width=150,
        )
        save_figure(figure, output, vector=True)
        plt.close(figure)
    return True
