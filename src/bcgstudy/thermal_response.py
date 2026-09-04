"""Paired outside-field thermal 8-13 Hz response for BCG method comparison.

The MRI simulator is an outside-field concordance reference for the
baseline-normalized posterior 8-13 Hz thermal change, not neural ground
truth. FASTR is the BCG-contaminated scanner reference. Correction arms
are compared on the same paired trials, with cardiac residual tabulated
separately.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import numpy as np

from bcg_correction.evaluation import band_integral, divide_or_nan
from bcg_correction.figure_style import (
    DASH,
    INK,
    MUTED,
    SPECTRA_SIZE,
    STYLE,
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
)
from bcgnet.compare.arms import CLEAN_ARMS

CORRECTED_METHODS = tuple(arm.key for arm in CLEAN_ARMS)
SCANNER_SERIES = ("fastr", *CORRECTED_METHODS)
METHOD_ORDER = ("simulator", *SCANNER_SERIES)
CANONICAL_EEG = (
    "Fp1",
    "Fp2",
    "F3",
    "F4",
    "C3",
    "C4",
    "P3",
    "P4",
    "O1",
    "O2",
    "F7",
    "F8",
    "T7",
    "T8",
    "P7",
    "P8",
    "Fz",
    "Cz",
    "Pz",
    "Oz",
    "FC1",
    "FC2",
    "CP1",
    "CP2",
    "FC5",
    "FC6",
    "CP5",
    "CP6",
    "TP9",
    "TP10",
    "POz",
    "F1",
    "F2",
    "C1",
    "C2",
    "P1",
    "P2",
    "AF3",
    "AF4",
    "FC3",
    "FC4",
    "CP3",
    "CP4",
    "PO3",
    "PO4",
    "F5",
    "F6",
    "C5",
    "C6",
    "P5",
    "P6",
    "AF7",
    "AF8",
    "FT7",
    "FT8",
    "TP7",
    "TP8",
    "PO7",
    "PO8",
    "FT9",
    "FT10",
    "FPz",
    "CPz",
)


@dataclass(frozen=True, slots=True)
class ThermalProtocol:
    runs: tuple[int, ...]
    n_triggers: int
    prestimulus_seconds: tuple[float, float]
    ramp_up_seconds: float
    hold_seconds: float
    posterior_channels: tuple[str, ...]
    band_hz: tuple[float, float]
    peak_to_peak_uv: float
    welch_nperseg_seconds: float

    @property
    def plateau_seconds(self) -> tuple[float, float]:
        return (self.ramp_up_seconds, self.ramp_up_seconds + self.hold_seconds)

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> ThermalProtocol:
        required = {
            "runs",
            "n_triggers",
            "prestimulus_seconds",
            "ramp_up_seconds",
            "hold_seconds",
            "posterior",
            "band_hz",
            "peak_to_peak_uv",
            "welch_nperseg_seconds",
        }
        missing = sorted(required - set(values))
        if missing:
            raise ValueError(f"missing thermal settings: {', '.join(missing)}")

        runs = tuple(int(run) for run in values["runs"])  # type: ignore[arg-type]
        prestimulus = tuple(float(value) for value in values["prestimulus_seconds"])  # type: ignore[arg-type]
        band = tuple(float(value) for value in values["band_hz"])  # type: ignore[arg-type]
        posterior = tuple(str(value) for value in values["posterior"])  # type: ignore[arg-type]
        if not runs or any(run < 1 for run in runs) or len(set(runs)) != len(runs):
            raise ValueError("runs must contain unique positive integers")
        if len(prestimulus) != 2 or prestimulus[0] >= prestimulus[1]:
            raise ValueError("prestimulus_seconds must be increasing")
        if len(band) != 2 or band[0] >= band[1]:
            raise ValueError("band_hz must be increasing")
        if not posterior or len(set(posterior)) != len(posterior):
            raise ValueError("posterior must contain unique channel names")

        protocol = cls(
            runs=runs,
            n_triggers=int(values["n_triggers"]),  # type: ignore[arg-type]
            prestimulus_seconds=(prestimulus[0], prestimulus[1]),
            ramp_up_seconds=float(values["ramp_up_seconds"]),  # type: ignore[arg-type]
            hold_seconds=float(values["hold_seconds"]),  # type: ignore[arg-type]
            posterior_channels=posterior,
            band_hz=(band[0], band[1]),
            peak_to_peak_uv=float(values["peak_to_peak_uv"]),  # type: ignore[arg-type]
            welch_nperseg_seconds=float(values["welch_nperseg_seconds"]),  # type: ignore[arg-type]
        )
        positive = (
            protocol.n_triggers,
            protocol.ramp_up_seconds,
            protocol.hold_seconds,
            protocol.peak_to_peak_uv,
            protocol.welch_nperseg_seconds,
        )
        if any(value <= 0 for value in positive):
            raise ValueError(
                "thermal counts, durations, and thresholds must be positive"
            )
        return protocol


@dataclass(frozen=True, slots=True)
class ThermalRecording:
    bids_id: str
    run: int
    name: str
    vhdr: Path
    n_trig_therm: int
    duration_seconds: float


def sequences_match(
    simulator: Sequence[tuple[float, str]],
    scanner: Sequence[tuple[float, str]],
) -> bool:
    if len(simulator) != len(scanner) or not simulator:
        return False
    return tuple(simulator) == tuple(scanner)


def eeg_channel_names(names: Sequence[str]) -> list[str]:
    return [name for name in names if name.strip().upper() != "ECG"]


def response_ratio_db(plateau_power: float, prestim_power: float) -> float:
    if (
        not np.isfinite(plateau_power)
        or not np.isfinite(prestim_power)
        or plateau_power <= 0
        or prestim_power <= 0
    ):
        return float("nan")
    return float(10.0 * np.log10(plateau_power / prestim_power))


def median_absolute_error(method_r: np.ndarray, simulator_r: np.ndarray) -> float:
    method_r = np.asarray(method_r, dtype=float)
    simulator_r = np.asarray(simulator_r, dtype=float)
    if method_r.shape != simulator_r.shape:
        raise ValueError("method and simulator responses must be paired")
    delta = np.abs(method_r - simulator_r)
    finite = np.isfinite(delta)
    if not np.any(finite):
        return float("nan")
    return float(np.median(delta[finite]))


def signed_median_difference(method_r: np.ndarray, simulator_r: np.ndarray) -> float:
    method_r = np.asarray(method_r, dtype=float)
    simulator_r = np.asarray(simulator_r, dtype=float)
    if method_r.shape != simulator_r.shape:
        raise ValueError("method and simulator responses must be paired")
    delta = method_r - simulator_r
    finite = np.isfinite(delta)
    if not np.any(finite):
        return float("nan")
    return float(np.median(delta[finite]))


def reference_keep_mask(
    simulator_complete: np.ndarray,
    simulator_quality: np.ndarray,
    scanner_complete: np.ndarray,
) -> np.ndarray:
    masks = tuple(
        np.asarray(mask, dtype=bool)
        for mask in (simulator_complete, simulator_quality, scanner_complete)
    )
    if len({mask.shape for mask in masks}) != 1:
        raise ValueError("paired trial masks must have the same shape")
    return masks[0] & masks[1] & masks[2]


def participant_method_summaries(trial_rows: Sequence[Mapping]) -> list[dict]:
    grouped: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    for row in trial_rows:
        grouped[(str(row["bids_id"]), str(row["method"]))].append(
            (float(row["r_method"]), float(row["r_simulator"]))
        )
    summaries = []
    for bids_id, method in sorted(grouped):
        method_r = np.array([pair[0] for pair in grouped[(bids_id, method)]])
        sim_r = np.array([pair[1] for pair in grouped[(bids_id, method)]])
        finite = np.isfinite(method_r) & np.isfinite(sim_r)
        summaries.append(
            {
                "bids_id": bids_id,
                "method": method,
                "n_trials": int(np.count_nonzero(finite)),
                "median_absolute_error": median_absolute_error(
                    method_r[finite], sim_r[finite]
                ),
                "signed_median_difference": signed_median_difference(
                    method_r[finite], sim_r[finite]
                ),
                "median_r_method": float(np.median(method_r[finite]))
                if np.any(finite)
                else float("nan"),
                "median_r_simulator": float(np.median(sim_r[finite]))
                if np.any(finite)
                else float("nan"),
            }
        )
    return summaries


def add_fastr_improvement(rows: Sequence[Mapping]) -> list[dict]:
    fastr = {
        str(row["bids_id"]): float(row["median_absolute_error"])
        for row in rows
        if row["method"] == "fastr"
    }
    subjects = {str(row["bids_id"]) for row in rows}
    missing = sorted(subjects - fastr.keys())
    if missing:
        raise ValueError(f"missing FASTR participant summaries: {', '.join(missing)}")
    return [
        {
            **dict(row),
            "improvement_vs_fastr": (
                fastr[str(row["bids_id"])] - float(row["median_absolute_error"])
            ),
        }
        for row in rows
    ]


def choose_completed_thermal(
    candidates: Sequence[ThermalRecording],
    *,
    expected_triggers: int,
) -> ThermalRecording | None:
    complete = [item for item in candidates if item.n_trig_therm == expected_triggers]
    if not complete:
        return None
    return max(complete, key=lambda item: (item.duration_seconds, item.name))


def _is_trig_therm(kind: str, description: str) -> bool:
    return "therm" in f"{kind} {description}".lower()


def trig_therm_samples(vmrk: Path) -> list[int]:
    samples: list[int] = []
    text = vmrk.read_text(encoding="latin-1")
    for line in text.splitlines():
        if not line.startswith("Mk") or "=" not in line:
            continue
        parts = line.split("=", 1)[1].split(",")
        if len(parts) < 3:
            continue
        if not _is_trig_therm(parts[0], parts[1]):
            continue
        position = int(parts[2]) - 1
        if position < 0:
            raise ValueError("Trig_therm sample is negative")
        samples.append(position)
    if any(later <= earlier for earlier, later in pairwise(samples)):
        raise ValueError("Trig_therm samples are non-increasing")
    return samples


def count_trig_therm(vmrk: Path) -> int:
    return len(trig_therm_samples(vmrk))


def read_trial_sequence(path: Path) -> list[tuple[float, str]]:
    text = path.read_text(encoding="latin-1")
    rows = list(csv.reader(text.splitlines()))
    if not rows:
        return []
    header = [column.strip().lower() for column in rows[0]]

    def _index(*names: str) -> int | None:
        for name in names:
            if name in header:
                return header.index(name)
        for name in names:
            for index, column in enumerate(header):
                if name in column:
                    return index
        return None

    i_temp = _index("stimulus_temp")
    i_surface = _index("selected_surface")
    if i_temp is None:
        return []
    sequence: list[tuple[float, str]] = []
    for row in rows[1:]:
        if i_temp >= len(row) or not row[i_temp].strip():
            continue
        try:
            temperature = round(float(row[i_temp]), 1)
        except ValueError:
            continue
        surface = (
            row[i_surface].strip()
            if i_surface is not None and i_surface < len(row)
            else ""
        )
        surface = "".join(character for character in surface if character.isdigit())
        sequence.append((temperature, surface))
    return sequence


def best_trial_summary(
    folder: Path,
    run: int,
    *,
    expected_trials: int,
) -> Path | None:
    if not folder.is_dir():
        return None
    matches = []
    for path in folder.glob("*TrialSummary.csv"):
        if path.name.startswith("._"):
            continue
        name = path.name.lower()
        if f"run{run}" not in name and f"run_{run}" not in name:
            continue
        sequence = read_trial_sequence(path)
        matches.append((len(sequence) == expected_trials, len(sequence), path))
    if not matches:
        return None
    matches.sort()
    return matches[-1][2]


def recording_duration_seconds(vhdr: Path) -> float:
    sampling_interval_us = None
    n_points = None
    text = vhdr.read_text(encoding="latin-1")
    for line in text.splitlines():
        if line.startswith("SamplingInterval="):
            sampling_interval_us = float(line.split("=", 1)[1])
        elif line.startswith("DataPoints="):
            n_points = float(line.split("=", 1)[1])
    if sampling_interval_us and n_points:
        return n_points * sampling_interval_us * 1e-6
    eeg = vhdr.with_suffix(".eeg")
    if eeg.is_file() and sampling_interval_us:
        n_samples = eeg.stat().st_size / (64 * 4)
        return n_samples * sampling_interval_us * 1e-6
    return float("nan")


def _welch_nfft(n_times: int, sfreq: float, nperseg_seconds: float) -> int:
    return min(round(sfreq * nperseg_seconds), n_times)


def _epochs_array(data: np.ndarray, sfreq: float):
    import mne

    n_channels = data.shape[1]
    info = mne.create_info(
        [f"EEG{index:02d}" for index in range(n_channels)],
        sfreq,
        ch_types="eeg",
    )
    return mne.EpochsArray(
        np.asarray(data, dtype=float), info, tmin=0.0, verbose="ERROR"
    )


def epoch_spectrum(
    data: np.ndarray,
    *,
    sfreq: float,
    nperseg_seconds: float,
    picks: Sequence[int] | None = None,
    fmin: float = 1.0,
    fmax: float = 40.0,
):
    """Per-epoch Welch PSD via ``Epochs.compute_psd``."""
    epochs = _epochs_array(data, sfreq)
    n_fft = _welch_nfft(data.shape[-1], sfreq, nperseg_seconds)
    spectrum = epochs.compute_psd(
        method="welch",
        fmin=fmin,
        fmax=fmax,
        n_fft=n_fft,
        n_overlap=n_fft // 2,
        picks=list(picks) if picks is not None else None,
        verbose="ERROR",
    )
    psd, freqs = spectrum.get_data(return_freqs=True)
    return freqs, psd


def epoch_band_power(
    data: np.ndarray,
    *,
    sfreq: float,
    band: tuple[float, float],
    nperseg_seconds: float,
    picks: Sequence[int] | None = None,
) -> np.ndarray:
    freqs, psd = epoch_spectrum(
        data,
        sfreq=sfreq,
        nperseg_seconds=nperseg_seconds,
        picks=picks,
        fmin=band[0],
        fmax=band[1],
    )
    integrated = band_integral(freqs, psd, band[0], band[1])
    return np.median(integrated, axis=-1)


def trial_keep_mask(
    epochs: np.ndarray,
    *,
    peak_to_peak_uv: float,
) -> np.ndarray:
    if epochs.ndim != 3:
        raise ValueError("epochs must be n_trials x n_channels x n_times")
    peak = np.ptp(epochs, axis=-1)
    limit = peak_to_peak_uv * 1e-6
    finite = np.all(np.isfinite(epochs), axis=(1, 2))
    return finite & np.all(peak < limit, axis=1)


def posterior_indices(
    ch_names: Sequence[str],
    posterior_channels: Sequence[str],
) -> list[int]:
    index = {name: i for i, name in enumerate(ch_names)}
    missing = [name for name in posterior_channels if name not in index]
    if missing:
        raise ValueError(f"missing posterior channels: {missing}")
    return [index[name] for name in posterior_channels]


def reorder_to_canonical(raw):
    if "ECG" in raw.ch_names:
        raw.set_channel_types({name: "ecg" for name in raw.ch_names if name == "ECG"})
    present = [name for name in CANONICAL_EEG if name in raw.ch_names]
    if len(present) != len(CANONICAL_EEG):
        missing = [name for name in CANONICAL_EEG if name not in raw.ch_names]
        raise ValueError(f"recording is missing EEG channels: {missing}")
    picks = list(present)
    if "ECG" in raw.ch_names:
        picks.append("ECG")
    return raw.pick(picks)


def preprocess_raw(raw, *, filter_hz: tuple[float, float]):
    raw = raw.copy()
    raw = reorder_to_canonical(raw)
    raw.filter(
        filter_hz[0],
        filter_hz[1],
        picks="eeg",
        fir_design="firwin",
        verbose="ERROR",
    )
    raw.set_eeg_reference("average", ch_type="eeg", verbose="ERROR")
    return raw


def extract_windows(
    raw,
    event_samples: Sequence[int],
    window: tuple[float, float],
    *,
    picks: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    sfreq = float(raw.info["sfreq"])
    n_times = raw.n_times
    tmin, tmax = window
    start_offset = round(tmin * sfreq)
    stop_offset = round(tmax * sfreq)
    width = stop_offset - start_offset
    data = raw.get_data(picks=list(picks))
    epochs = []
    keep = []
    for sample in event_samples:
        start = int(sample) + start_offset
        stop = int(sample) + stop_offset
        ok = start >= 0 and stop <= n_times and width > 0
        keep.append(ok)
        if ok:
            epochs.append(data[:, start:stop])
        else:
            epochs.append(np.zeros((len(picks), max(width, 1))))
    return np.stack(epochs, axis=0), np.asarray(keep, dtype=bool)


def cardiac_residual_from_summary(
    rows: Sequence[Mapping[str, str]],
    *,
    bids_id: str,
    method: str,
    runs: Sequence[int],
    column: str | None = None,
) -> float:
    field = column or f"local_5_ecg_regressed_ratio_{method}"
    wanted = {int(run) for run in runs}
    values = []
    for row in rows:
        if row.get("bids_id") != bids_id:
            continue
        run = row.get("run")
        if run in (None, "", "None"):
            continue
        if int(run) not in wanted:
            continue
        try:
            value = float(row[field])
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(value):
            values.append(value)
    if not values:
        raise ValueError(
            f"missing cardiac residual for {bids_id} {method} runs {sorted(wanted)}"
        )
    return float(np.median(values))


def _series_style(key: str) -> tuple[str, str, str]:
    if key == "simulator":
        return "simulator", INK, "X"
    if key == "fastr":
        return "FASTR", UNCORRECTED, UNCORRECTED_MARKER
    return arm_label(key), arm_color(key), arm_marker(key)


def _protocol_caption(protocol: ThermalProtocol) -> str:
    channels = " ".join(protocol.posterior_channels)
    band = f"{protocol.band_hz[0]:g}-{protocol.band_hz[1]:g}"
    pre0, pre1 = protocol.prestimulus_seconds
    plat0, plat1 = protocol.plateau_seconds
    return (
        f"R(f) = 10 log10(P_plateau / P_prestim) on fixed posterior channels "
        f"({channels}). Prestimulus is {pre0:g} to {pre1:g} s relative to "
        f"Trig_therm; plateau is the documented hold ({plat0:g}-{plat1:g} s), "
        f"not a window chosen from the EEG. Traces are the median across "
        f"participants of each participant's trial-median Welch PSD. The "
        f"shaded band is the simulator participant IQR when n >= 3. Simulator "
        f"EEG has ordinary preprocessing and no BCG correction. FASTR is "
        f"gradient-corrected but BCG-uncorrected. Scanner arms are then AAS, "
        f"PCA-OBS, or BCGNet; BCGNet includes training-time samples written "
        f"back to the full recording. The {band} Hz band is the analysis band, "
        f"not labelled alpha: it is also BCG harmonics {band}. This page does "
        f"not rank methods on absolute power. Cardiac-locked residual and "
        f"improvement versus FASTR are separate tables."
    )


def plot_response_spectra(
    curves: Mapping[str, Mapping],
    output: Path,
    protocol: ThermalProtocol,
):
    """R(f) for the simulator, FASTR, and each corrected method."""
    import matplotlib.pyplot as plt

    with plt.rc_context(STYLE):
        fig, axis = plt.subplots(figsize=(7.24, 4.10), layout="constrained")
        fig.get_layout_engine().set(w_pad=0.02, h_pad=0.04)
        entries = []
        for key in METHOD_ORDER:
            if key not in curves:
                continue
            curve = curves[key]
            freq = np.asarray(curve["freq"], dtype=float)
            median = np.asarray(curve["median"], dtype=float)
            label, color, marker = _series_style(key)
            if key == "simulator":
                fill_iqr(
                    axis, freq, curve.get("q1"), curve.get("q3"), color, alpha=0.14
                )
            axis.plot(
                freq,
                median,
                color=color,
                lw=1.7 if key == "simulator" else 1.15,
                ls=DASH if key == "fastr" else "-",
                zorder=3.4 if key == "simulator" else 2.7,
                solid_capstyle="round",
            )
            entries.append((label, color, marker))
        axis.axhline(0.0, color=MUTED, lw=0.6, ls=":", zorder=1)
        axis.set_ylabel(r"$R(f)$ (dB)")
        frequency_axis(axis, max_hz=40.0)
        axis.set_xlim(1.0, 40.0)
        low, high = axis.get_ylim()
        axis.set_ylim(low, high + 0.22 * (high - low))
        band = f"{protocol.band_hz[0]:g}-{protocol.band_hz[1]:g}"
        panel(
            axis,
            "A",
            f"Thermal {band} Hz spectral response",
            "outside-field simulator vs FASTR vs corrected scanner EEG",
        )
        arm_legend(fig, entries)
        fig.suptitle(
            f"Paired thermal response, posterior {band} Hz",
            fontsize=9.0,
            fontweight="bold",
            x=0.0,
            ha="left",
            color=INK,
        )
        figure_caption(fig, _protocol_caption(protocol))
        save_figure(fig, output, vector=True)
        return fig


def plot_prestim_plateau_spectra(
    panels: Mapping[str, Mapping],
    output: Path,
    protocol: ThermalProtocol,
):
    """Allen-style prestimulus vs plateau PSD, one panel per series."""
    import matplotlib.pyplot as plt

    keys = [key for key in METHOD_ORDER if key in panels]
    if not keys:
        raise ValueError("no spectra to plot")
    n_columns = 3
    n_rows = int(np.ceil(len(keys) / n_columns))
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(
            n_rows,
            n_columns,
            figsize=SPECTRA_SIZE,
            sharex=True,
            sharey=True,
            layout="constrained",
        )
        fig.get_layout_engine().set(w_pad=0.04, h_pad=0.06, hspace=0.14, wspace=0.06)
        flat_axes = np.asarray(axes).ravel()
        letters = "ABCDEF"
        for axis, key, letter in zip(flat_axes, keys, letters, strict=False):
            spec = panels[key]
            freq = np.asarray(spec["freq"], dtype=float)
            prestim = np.asarray(spec["prestim"], dtype=float)
            plateau = np.asarray(spec["plateau"], dtype=float)
            _label, color, _marker = _series_style(key)
            axis.plot(
                freq,
                np.where(prestim > 0, prestim, np.nan),
                color=color,
                lw=1.05,
                ls=DASH,
                zorder=2.5,
            )
            axis.plot(
                freq,
                np.where(plateau > 0, plateau, np.nan),
                color=color,
                lw=1.3,
                zorder=2.8,
                solid_capstyle="round",
            )
            axis.set_yscale("log")
            frequency_axis(axis, max_hz=40.0)
            title, _color, _marker = _series_style(key)
            if key == "simulator":
                qualifier = "outside field, no BCG correction"
            elif key == "fastr":
                qualifier = "scanner, gradient-corrected, no BCG correction"
            else:
                qualifier = "scanner, after BCG correction"
            panel(axis, letter, title, qualifier)
            spec = axis.get_subplotspec()
            if spec is not None and spec.is_first_col():
                axis.set_ylabel(r"PSD ($\mu\mathrm{V}^2/\mathrm{Hz}$)")
        for axis in flat_axes[len(keys) :]:
            axis.set_visible(False)
        pre_s = abs(protocol.prestimulus_seconds[0])
        hold_s = protocol.hold_seconds
        legend_axis = flat_axes[1] if flat_axes[1].get_visible() else flat_axes[0]
        linestyle_key(
            legend_axis,
            [
                (DASH, f"prestimulus ({pre_s:g} s)"),
                ("-", f"plateau ({hold_s:g} s hold)"),
            ],
            loc="upper right",
        )
        fig.suptitle(
            "Posterior prestimulus and plateau spectra",
            fontsize=9.0,
            fontweight="bold",
            x=0.0,
            ha="left",
            color=INK,
        )
        plat0, plat1 = protocol.plateau_seconds
        figure_caption(
            fig,
            "Median across participants of the trial-median posterior Welch PSD. "
            f"Dashed is prestimulus; solid is the documented hold ({plat0:g}-"
            f"{plat1:g} s). Absolute plateau power is an Allen-style descriptive "
            "check and is not used to rank methods: visit impedance and cap "
            "placement shift the level. FASTR is gradient-corrected but "
            "BCG-uncorrected. The quantity used for comparison is the "
            "baseline-normalized 8-13 Hz response in the companion spectrum and "
            "in participant_summary.csv.",
        )
        save_figure(fig, output, vector=True)
        return fig


def median_spectrum(psds: np.ndarray, keep: np.ndarray) -> np.ndarray:
    if not np.any(keep):
        return np.full(psds.shape[-1], np.nan)
    return np.median(psds[keep], axis=(0, 1))


def response_spectrum_db(
    plateau_psd: np.ndarray, prestim_psd: np.ndarray
) -> np.ndarray:
    ratio = divide_or_nan(plateau_psd, prestim_psd)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 10.0 * np.log10(ratio)
