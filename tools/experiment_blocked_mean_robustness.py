"""Test cross-fit granularity and known-signal transfer on holdout recordings."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path

import mne
import numpy as np
import yaml

from bcg_correction.adaptive import correct_cross_fitted_reference_mean
from bcg_correction.bcg import rr_gap_spans
from bcg_correction.cardiac import detect_r_peaks
from bcg_correction.correction_report import compute_correction_profile
from bcg_correction.metrics import (
    delay_estimation_eeg,
    estimate_ecg_to_bcg_delay,
    incremental_signal_transfer,
    is_posterior_eeg_channel,
)
from bcgnet.compare.config import load_compare_config

_CONFIG_KEYS = frozenset(
    {
        "compare_config",
        "holdout_metrics",
        "output_root",
        "cross_fit_fold_counts",
        "injection_amplitude_uv",
        "event_count",
        "seed",
    }
)


@dataclass(frozen=True, slots=True)
class RobustnessConfig:
    """Strict settings for fold sensitivity and signal transfer."""

    compare_config: Path
    holdout_metrics: Path
    output_root: Path
    cross_fit_fold_counts: tuple[int, ...]
    injection_amplitude_uv: float
    event_count: int
    seed: int


@dataclass(frozen=True, slots=True)
class Recording:
    """One unique FASTR recording selected from the completed holdout."""

    subject: str
    label: str
    path: Path


def _load_config(path: Path) -> RobustnessConfig:
    config_path = path.expanduser().resolve()
    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or frozenset(document) != _CONFIG_KEYS:
        raise ValueError(
            "robustness configuration must contain exactly: "
            + ", ".join(sorted(_CONFIG_KEYS))
        )
    base = config_path.parent
    folds = document["cross_fit_fold_counts"]
    if (
        not isinstance(folds, list)
        or len(folds) < 2
        or any(
            isinstance(value, bool)
            or not isinstance(value, Integral)
            or value < 2
            for value in folds
        )
    ):
        raise ValueError(
            "cross_fit_fold_counts must contain at least two integers >= 2"
        )
    if len(set(folds)) != len(folds):
        raise ValueError("cross_fit_fold_counts must be unique")
    return RobustnessConfig(
        compare_config=_resolve_path(document["compare_config"], base),
        holdout_metrics=_resolve_path(document["holdout_metrics"], base),
        output_root=_resolve_path(document["output_root"], base),
        cross_fit_fold_counts=tuple(int(value) for value in folds),
        injection_amplitude_uv=_positive_number(
            document["injection_amplitude_uv"],
            name="injection_amplitude_uv",
        ),
        event_count=_integer(document["event_count"], name="event_count", minimum=1),
        seed=_integer(document["seed"], name="seed", minimum=0),
    )


def _resolve_path(value: object, base: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("configured paths must be nonempty strings")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _positive_number(value: object, *, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        or value <= 0.0
    ):
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _integer(value: object, *, name: str, minimum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or value < minimum
    ):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _make_injections(
    *,
    channel_count: int,
    sample_count: int,
    sampling_rate: float,
    amplitude_uv: float,
    event_count: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Return deterministic, rank-one continuous and transient EEG signals."""
    if min(channel_count, sample_count, event_count) < 1:
        raise ValueError("injection dimensions and event_count must be positive")
    generator = np.random.default_rng(seed)
    spatial = generator.normal(size=channel_count)
    spatial /= np.max(np.abs(spatial))
    amplitude = float(amplitude_uv) * 1e-6
    times = np.arange(sample_count, dtype=np.float64) / sampling_rate
    tone = amplitude * np.sin(2.0 * np.pi * 10.0 * times)

    event_span = max(5, round(0.12 * sampling_rate))
    if event_span >= sample_count:
        raise ValueError("recording is too short for sparse injections")
    phase = np.linspace(0.0, 2.0 * np.pi, event_span, endpoint=False)
    waveform = np.sin(phase) * np.hanning(event_span)
    waveform /= np.max(np.abs(waveform))
    sparse = np.zeros(sample_count, dtype=np.float64)
    onsets = generator.integers(0, sample_count - event_span, size=event_count)
    signs = generator.choice((-1.0, 1.0), size=event_count)
    for onset, sign in zip(onsets, signs, strict=True):
        sparse[onset : onset + event_span] += sign * amplitude * waveform
    return {
        "tone_10hz": spatial[:, np.newaxis] * tone,
        "sparse_events": spatial[:, np.newaxis] * sparse,
    }



def _read_recordings(path: Path) -> tuple[Recording, ...]:
    with path.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    recordings = [
        Recording(
            subject=row["subject"],
            label=row["recording"],
            path=Path(row["input_vhdr"]).expanduser().resolve(),
        )
        for row in rows
        if row["method"] == "blocked_mean"
    ]
    keys = [(item.subject, item.label) for item in recordings]
    if not recordings or len(set(keys)) != len(keys):
        raise ValueError("holdout metrics must define unique blocked-mean recordings")
    missing = [str(item.path) for item in recordings if not item.path.is_file()]
    if missing:
        raise FileNotFoundError("missing FASTR recordings: " + ", ".join(missing))
    return tuple(recordings)


def _read_posterior_recording(
    path: Path,
) -> tuple[np.ndarray, tuple[str, ...], float]:
    raw = mne.io.read_raw_brainvision(path, preload=False, verbose="ERROR")
    try:
        posterior = tuple(
            name for name in raw.ch_names if is_posterior_eeg_channel(name)
        )
        if not posterior or "ECG" not in raw.ch_names:
            raise RuntimeError(f"posterior EEG or ECG is missing: {path}")
        names = (*posterior, "ECG")
        return raw.get_data(picks=list(names)), names, float(raw.info["sfreq"])
    finally:
        raw.close()


def _median(rows: list[dict[str, object]], key: str) -> float:
    return float(np.median([float(row[key]) for row in rows]))


def _summarize(
    quality_rows: list[dict[str, object]],
    transfer_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    quality_by_fold: dict[int, list[dict[str, object]]] = defaultdict(list)
    transfer_by_fold: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in quality_rows:
        quality_by_fold[int(row["fold_count"])].append(row)
    for row in transfer_rows:
        transfer_by_fold[int(row["fold_count"])].append(row)
    summaries = []
    for fold_count in quality_by_fold:
        quality = quality_by_fold[fold_count]
        transfer = transfer_by_fold[fold_count]
        summaries.append(
            {
                "fold_count": fold_count,
                "recording_count": len(quality),
                "locked_ratio_median": _median(quality, "locked_ratio"),
                "specificity_median": _median(quality, "specificity"),
                "alpha_collateral_fraction_median": _median(
                    quality,
                    "alpha_collateral_fraction",
                ),
                "transfer_relative_error_median": _median(
                    transfer,
                    "relative_error",
                ),
                "transfer_gain_absolute_error_median": float(
                    np.median([abs(float(row["gain"]) - 1.0) for row in transfer])
                ),
                "transfer_cosine_similarity_median": _median(
                    transfer,
                    "cosine_similarity",
                ),
            }
        )
    return summaries


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError("refusing to write an empty result table")
    with path.open("x", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    experiment = _load_config(arguments.config)
    comparison = load_compare_config(experiment.compare_config)
    recordings = _read_recordings(experiment.holdout_metrics)
    experiment.output_root.mkdir(parents=True, exist_ok=False)

    quality_rows: list[dict[str, object]] = []
    transfer_rows: list[dict[str, object]] = []
    for recording_index, recording in enumerate(recordings):
        data, names, sampling_rate = _read_posterior_recording(recording.path)
        ecg_index = len(names) - 1
        detection = detect_r_peaks(
            data[ecg_index],
            sampling_rate,
            config=comparison.correction.detector,
        )
        delay = estimate_ecg_to_bcg_delay(
            delay_estimation_eeg(data, names, ecg_channel_index=ecg_index),
            detection.peak_samples,
            sampling_rate_hz=sampling_rate,
        ).best_delay_seconds
        gaps = rr_gap_spans(
            detection.peak_samples,
            sampling_rate,
            comparison.correction.detector.maximum_rr_seconds,
        )
        gap_fraction = sum(stop - start for start, stop in gaps) / data.shape[1]
        injections = _make_injections(
            channel_count=ecg_index,
            sample_count=data.shape[1],
            sampling_rate=sampling_rate,
            amplitude_uv=experiment.injection_amplitude_uv,
            event_count=experiment.event_count,
            seed=experiment.seed + recording_index,
        )
        common = {
            "subject": recording.subject,
            "recording": recording.label,
            "input_vhdr": str(recording.path),
            "detector_status": detection.quality.status,
        }
        for fold_count in experiment.cross_fit_fold_counts:
            corrected = correct_cross_fitted_reference_mean(
                data,
                ecg_channel_index=ecg_index,
                peak_samples=detection.peak_samples,
                sampling_rate_hz=sampling_rate,
                delay_seconds=delay,
                window_seconds=comparison.correction.window_seconds,
                fold_count=fold_count,
            )
            profile = compute_correction_profile(
                data,
                corrected,
                names,
                ecg_channel_index=ecg_index,
                peak_samples=detection.peak_samples,
                sampling_rate_hz=sampling_rate,
                delay_seconds=delay,
                window_seconds=comparison.correction.window_seconds,
                gap_fraction=gap_fraction,
                method=f"blocked_mean_fold_{fold_count}",
                label=recording.label,
            )
            if profile is None:
                raise RuntimeError(f"too few beats for {recording.path}")
            quality_rows.append(
                common
                | {
                    "fold_count": fold_count,
                    "locked_ratio": profile.locked_ratio,
                    "specificity": profile.specificity,
                    "alpha_collateral_fraction": (
                        profile.alpha_collateral_fraction
                    ),
                }
            )
            for injection_name, injection in injections.items():
                injected_data = data.copy()
                injected_data[:ecg_index] += injection
                injected_corrected = correct_cross_fitted_reference_mean(
                    injected_data,
                    ecg_channel_index=ecg_index,
                    peak_samples=detection.peak_samples,
                    sampling_rate_hz=sampling_rate,
                    delay_seconds=delay,
                    window_seconds=comparison.correction.window_seconds,
                    fold_count=fold_count,
                )
                transfer = incremental_signal_transfer(
                    injection,
                    injected_corrected[:ecg_index] - corrected[:ecg_index],
                )
                transfer_rows.append(
                    common
                    | {
                        "fold_count": fold_count,
                        "injection": injection_name,
                        "gain": transfer.gain,
                        "relative_error": transfer.relative_error,
                        "cosine_similarity": transfer.cosine_similarity,
                    }
                )
                del injected_data, injected_corrected
            del corrected
        print(
            f"{recording.subject} {recording.label}: "
            f"{len(experiment.cross_fit_fold_counts)} fold counts",
            flush=True,
        )
    _write_csv(experiment.output_root / "fold_quality.csv", quality_rows)
    _write_csv(experiment.output_root / "fold_transfer.csv", transfer_rows)
    _write_csv(
        experiment.output_root / "fold_summary.csv",
        _summarize(quality_rows, transfer_rows),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
