"""Participant-balanced descriptive summaries of schema-2 BCG audit profiles.

These existing audit outputs are not made independent holdouts by changing the
metrics. There is no Pareto ranking based on phase locking or a preferred arm.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

from bcg_correction.correction_report import participant_values, read_profile
from bcgnet.compare.arms import CLEAN_ARMS

# Preserve the original audit's deterministic resampling protocol.
_BOOTSTRAP_RESAMPLES = 20_000
_BOOTSTRAP_SEED = 20260831


def _bootstrap_median_interval(values):
    samples = np.asarray(values, dtype=float)
    if samples.ndim != 1 or samples.size < 2 or not np.all(np.isfinite(samples)):
        raise ValueError("bootstrap needs at least two finite participant differences")
    rng = np.random.default_rng(_BOOTSTRAP_SEED)
    resampled = rng.choice(samples, size=(_BOOTSTRAP_RESAMPLES, samples.size))
    low, high = np.percentile(np.median(resampled, axis=1), [2.5, 97.5])
    return float(low), float(high)


def _metrics(rows):
    return [
        key for key in rows[0] if key.startswith("local_") and key.endswith("_ratio")
    ]


def _lookup(rows):
    lookup = {(row["subject"], row["recording"], row["method"]): row for row in rows}
    if len(lookup) != len(rows):
        raise ValueError("duplicate audit recording/method")
    methods = [
        arm.key for arm in CLEAN_ARMS if any(r["method"] == arm.key for r in rows)
    ]
    if set(methods) != {row["method"] for row in rows}:
        raise ValueError("unknown audit method")
    identities = [{(s, r) for s, r, m in lookup if m == method} for method in methods]
    if any(keys != identities[0] for keys in identities[1:]):
        raise ValueError("audit methods must have identical paired recordings")
    return lookup, methods, sorted(identities[0])


def _read_metrics(path):
    with path.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    if not rows or not _metrics(rows):
        raise ValueError(
            "missing metric columns: schema-2 local residuals required; "
            "regenerate audit outputs"
        )
    _lookup(rows)
    for row in rows:
        for key in _metrics(rows):
            row[key] = None if row[key] == "" else float(row[key])
            if row[key] is not None and not np.isfinite(row[key]):
                raise ValueError(f"nonfinite {key}")
    return rows


def _stats(values):
    if any(value is None or not np.isfinite(value) for value in values):
        return {"status": "unavailable", "median": None, "q1": None, "q3": None}
    q1, median, q3 = np.percentile(values, [25, 50, 75])
    return {
        "status": "descriptive",
        "median": float(median),
        "q1": float(q1),
        "q3": float(q3),
    }


def _participant_means(values):
    return [
        None if any(v is None for v in items) else float(np.mean(items))
        for _, items in sorted(values.items())
    ]


def _subject_paired_differences(rows, *, metric, first, second):
    lookup, _, identities = _lookup(rows)
    participants = defaultdict(list)
    for subject, recording in identities:
        a = lookup[subject, recording, first][metric]
        b = lookup[subject, recording, second][metric]
        participants[subject].append(None if a is None or b is None else a - b)
    return np.asarray(_participant_means(participants), dtype=float)


def _method_summary(rows):
    _, methods, _ = _lookup(rows)
    summaries = []
    for method in methods:
        selected = [row for row in rows if row["method"] == method]
        for metric in _metrics(rows):
            participants = defaultdict(list)
            for row in selected:
                participants[row["subject"]].append(row[metric])
            summaries.append(
                {
                    "method": method,
                    "metric": metric,
                    "recordings": len(selected),
                    "participants": len(participants),
                    **_stats(_participant_means(participants)),
                }
            )
    return summaries


def _paired_summary(rows):
    _, methods, _ = _lookup(rows)
    summaries = []
    for first, second in combinations(methods, 2):
        for metric in _metrics(rows):
            differences = _subject_paired_differences(
                rows, metric=metric, first=first, second=second
            )
            low, high = (None, None)
            if len(differences) >= 2 and np.all(np.isfinite(differences)):
                low, high = _bootstrap_median_interval(differences)
            summaries.append(
                {
                    "first": first,
                    "second": second,
                    "metric": metric,
                    "difference": "first_minus_second",
                    "participants": len(differences),
                    **_stats(differences),
                    "exploratory_unadjusted_ci95_low": low,
                    "exploratory_unadjusted_ci95_high": high,
                }
            )
    return summaries


def _spatial_summary(root):
    groups = defaultdict(list)
    for path in sorted(root.rglob("*_profile.npz")):
        profile = read_profile(path)
        groups[profile.method].append(profile)
    rows = []
    for arm in CLEAN_ARMS:
        profiles = groups[arm.key]
        if not profiles:
            continue
        before = np.median(participant_values(profiles, "topo_before"), axis=0)
        after = np.median(participant_values(profiles, "topo_after"), axis=0)
        for resolution, count in enumerate(profiles[0].block_counts):
            for channel, name in enumerate(profiles[0].channel_names):
                a, b = before[resolution, channel], after[resolution, channel]
                rows.append(
                    {
                        "method": arm.key,
                        "blocks": int(count),
                        "channel": str(name),
                        "before_uv": float(a) if np.isfinite(a) else None,
                        "after_uv": float(b) if np.isfinite(b) else None,
                    }
                )
    return rows


def _write_csv(path, rows):
    if not rows:
        raise ValueError("refusing to write an empty summary")
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    root = parser.parse_args(argv).results.expanduser().resolve()
    rows = _read_metrics(root / "holdout_metrics.csv")
    _write_csv(root / "method_summary.csv", _method_summary(rows))
    paired = _paired_summary(rows)
    if paired:
        _write_csv(root / "paired_comparisons.csv", paired)
    _write_csv(root / "spatial_summary.csv", _spatial_summary(root / "profiles"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
