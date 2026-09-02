"""Summarize paired and spatial evidence from a held-out BCG validation."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from bcg_correction.correction_report import read_profile
from bcgnet.compare.arms import BLOCKED_MEAN, CLEAN_ARMS

_FOCUS_METHOD = BLOCKED_MEAN.key
_METHODS = tuple(arm.key for arm in CLEAN_ARMS)
_COMPARATORS = tuple(method for method in _METHODS if method != _FOCUS_METHOD)
_METRICS = (
    ("locked_ratio", "negative"),
    # The same ratio on the EEG as written, with no ECG regression: the only
    # one of these that can see ECG-shaped residual left in the file.
    ("locked_ratio_raw", "negative"),
    ("specificity", "positive"),
    ("alpha_collateral_fraction", "negative"),
)
_BOOTSTRAP_RESAMPLES = 20_000
_BOOTSTRAP_SEED = 20260831


def _subject_paired_differences(
    rows: list[dict[str, object]],
    *,
    metric: str,
    comparator: str,
) -> np.ndarray:
    lookup = {
        (str(row["subject"]), str(row["recording"]), str(row["method"])): float(
            row[metric]
        )
        for row in rows
    }
    recordings = sorted(
        {
            (str(row["subject"]), str(row["recording"]))
            for row in rows
            if row["method"] == _FOCUS_METHOD
        }
    )
    by_subject: dict[str, list[float]] = defaultdict(list)
    for subject, recording in recordings:
        blocked = lookup[subject, recording, _FOCUS_METHOD]
        comparison = lookup[subject, recording, comparator]
        by_subject[subject].append(blocked - comparison)
    return np.asarray(
        [np.median(by_subject[subject]) for subject in sorted(by_subject)],
        dtype=np.float64,
    )


def _bootstrap_median_interval(values: np.ndarray) -> tuple[float, float]:
    samples = np.asarray(values, dtype=np.float64)
    if samples.ndim != 1 or samples.size < 2 or not np.all(np.isfinite(samples)):
        raise ValueError("bootstrap values must contain at least two finite values")
    generator = np.random.default_rng(_BOOTSTRAP_SEED)
    indices = generator.integers(
        0,
        samples.size,
        size=(_BOOTSTRAP_RESAMPLES, samples.size),
    )
    bootstrap = np.median(samples[indices], axis=1)
    low, high = np.quantile(bootstrap, (0.025, 0.975))
    return float(low), float(high)



def _read_metrics(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8", newline="") as source:
        rows: list[dict[str, object]] = list(csv.DictReader(source))
    required = {"subject", "recording", "method", "detector_status"}
    if not rows or any(not required.issubset(row) for row in rows):
        raise ValueError("holdout metric table is empty or incomplete")
    missing_metrics = [
        metric for metric, _direction in _METRICS if metric not in rows[0]
    ]
    if missing_metrics:
        raise ValueError(
            "holdout metric table is missing metric columns: "
            + ", ".join(missing_metrics)
        )
    for row in rows:
        for metric, _direction in _METRICS:
            value = float(row[metric])
            if not math.isfinite(value):
                raise ValueError(f"nonfinite {metric} in holdout metric table")
            row[metric] = value
    counts = defaultdict(int)
    for row in rows:
        counts[str(row["method"])] += 1
    if set(counts) != set(_METHODS) or len(set(counts.values())) != 1:
        raise ValueError("holdout methods must have equal paired row counts")
    return rows


def _method_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summaries = []
    for stratum in ("all", "ok", "degraded"):
        for method in _METHODS:
            selected = [
                row
                for row in rows
                if row["method"] == method
                and (stratum == "all" or row["detector_status"] == stratum)
            ]
            if not selected:
                continue
            summary: dict[str, object] = {
                "detector_stratum": stratum,
                "method": method,
                "recording_count": len(selected),
                "participant_count": len({row["subject"] for row in selected}),
            }
            for metric, _direction in _METRICS:
                values = np.asarray([row[metric] for row in selected], dtype=float)
                summary[f"{metric}_median"] = float(np.median(values))
                summary[f"{metric}_q1"] = float(np.quantile(values, 0.25))
                summary[f"{metric}_q3"] = float(np.quantile(values, 0.75))
            summaries.append(summary)
    return summaries


def _paired_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    lookup = {
        (str(row["subject"]), str(row["recording"]), str(row["method"])): row
        for row in rows
    }
    recordings = sorted(
        {
            (str(row["subject"]), str(row["recording"]))
            for row in rows
            if row["method"] == _FOCUS_METHOD
        }
    )
    summaries = []
    for comparator in _COMPARATORS:
        for metric, favorable_direction in _METRICS:
            differences = np.asarray(
                [
                    float(lookup[subject, recording, _FOCUS_METHOD][metric])
                    - float(lookup[subject, recording, comparator][metric])
                    for subject, recording in recordings
                ]
            )
            participant_differences = _subject_paired_differences(
                rows,
                metric=metric,
                comparator=comparator,
            )
            low, high = _bootstrap_median_interval(participant_differences)
            favorable = (
                differences < 0.0
                if favorable_direction == "negative"
                else differences > 0.0
            )
            summaries.append(
                {
                    "comparator": comparator,
                    "metric": metric,
                    "difference": "blocked_mean_minus_comparator",
                    "favorable_direction": favorable_direction,
                    "recording_median_difference": float(np.median(differences)),
                    "participant_median_difference": float(
                        np.median(participant_differences)
                    ),
                    "participant_bootstrap_low": low,
                    "participant_bootstrap_high": high,
                    "favorable_recording_count": int(np.count_nonzero(favorable)),
                    "recording_count": differences.size,
                    "participant_count": participant_differences.size,
                }
            )
    return summaries


def _pareto_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    lookup = {
        (str(row["subject"]), str(row["recording"]), str(row["method"])): row
        for row in rows
    }
    recordings = sorted(
        {
            (str(row["subject"]), str(row["recording"]))
            for row in rows
            if row["method"] == _FOCUS_METHOD
        }
    )
    summaries = []
    for comparator in _COMPARATORS:
        blocked_dominates = 0
        comparator_dominates = 0
        for subject, recording in recordings:
            blocked = lookup[subject, recording, _FOCUS_METHOD]
            other = lookup[subject, recording, comparator]
            blocked_values = np.asarray(
                [
                    blocked["locked_ratio"],
                    -float(blocked["specificity"]),
                    blocked["alpha_collateral_fraction"],
                ],
                dtype=float,
            )
            other_values = np.asarray(
                [
                    other["locked_ratio"],
                    -float(other["specificity"]),
                    other["alpha_collateral_fraction"],
                ],
                dtype=float,
            )
            blocked_dominates += int(
                np.all(blocked_values <= other_values)
                and np.any(blocked_values < other_values)
            )
            comparator_dominates += int(
                np.all(other_values <= blocked_values)
                and np.any(other_values < blocked_values)
            )
        summaries.append(
            {
                "comparator": comparator,
                "blocked_mean_dominates": blocked_dominates,
                "comparator_dominates": comparator_dominates,
                "tradeoff": len(recordings) - blocked_dominates - comparator_dominates,
                "recording_count": len(recordings),
            }
        )
    return summaries


def _correlation(first: np.ndarray, second: np.ndarray) -> float:
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1 or not np.all(np.isfinite(left)):
        raise ValueError("topography arrays must be matching finite vectors")
    left = left - left.mean()
    right = right - right.mean()
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(np.dot(left, right) / denominator) if denominator else 0.0


def _relative_error(reference: np.ndarray, estimate: np.ndarray) -> float:
    denominator = np.linalg.norm(reference)
    if denominator == 0.0:
        raise ValueError("artifact topography has zero energy")
    return float(np.linalg.norm(estimate - reference) / denominator)


def _median_fraction(numerator: np.ndarray, denominator: np.ndarray) -> float:
    valid = denominator > 0.0
    if not np.any(valid):
        raise ValueError("alpha topography has no positive channel")
    return float(np.median(numerator[valid] / denominator[valid]))


def _spatial_summary(profile_root: Path) -> list[dict[str, object]]:
    values: dict[str, dict[str, list[float]]] = {
        method: defaultdict(list) for method in _METHODS
    }
    paths = sorted(profile_root.rglob("*_profile.npz"))
    if not paths:
        raise ValueError("no holdout profiles were found")
    for path in paths:
        profile = read_profile(path)
        if profile.method not in values:
            raise ValueError(f"unknown profile method: {profile.method}")
        spatial = values[profile.method]
        spatial["artifact_removed_correlation"].append(
            _correlation(profile.topo_artifact, profile.topo_removed_locked)
        )
        spatial["artifact_map_relative_error"].append(
            _relative_error(profile.topo_artifact, profile.topo_removed_locked)
        )
        spatial["alpha_collateral_correlation"].append(
            _correlation(profile.topo_alpha_present, profile.topo_collateral_alpha)
        )
        spatial["all_channel_alpha_collateral_fraction"].append(
            _median_fraction(
                profile.topo_collateral_alpha,
                profile.topo_alpha_present,
            )
        )
    summaries = []
    for method in _METHODS:
        method_values = values[method]
        summary: dict[str, object] = {
            "method": method,
            "recording_count": len(method_values["artifact_removed_correlation"]),
        }
        for metric, observations in method_values.items():
            array = np.asarray(observations)
            summary[f"{metric}_median"] = float(np.median(array))
            summary[f"{metric}_q1"] = float(np.quantile(array, 0.25))
            summary[f"{metric}_q3"] = float(np.quantile(array, 0.75))
        summaries.append(summary)
    return summaries


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("refusing to write an empty summary")
    with path.open("x", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    root = arguments.results.expanduser().resolve()
    rows = _read_metrics(root / "holdout_metrics.csv")
    _write_csv(root / "method_summary.csv", _method_summary(rows))
    _write_csv(root / "paired_comparisons.csv", _paired_summary(rows))
    _write_csv(root / "pareto_comparisons.csv", _pareto_summary(rows))
    _write_csv(root / "spatial_summary.csv", _spatial_summary(root / "profiles"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
