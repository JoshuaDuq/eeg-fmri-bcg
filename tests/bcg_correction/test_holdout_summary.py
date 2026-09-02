import csv
import runpy
from pathlib import Path

import numpy as np


def test_subject_differences_give_each_participant_one_value() -> None:
    script = Path(__file__).parents[2] / "tools" / "summarize_holdout.py"
    differences = runpy.run_path(str(script))["_subject_paired_differences"]
    rows = [
        {"subject": "sub-1", "recording": "a", "method": "blocked_mean", "x": 1.0},
        {"subject": "sub-1", "recording": "a", "method": "aas", "x": 3.0},
        {"subject": "sub-1", "recording": "b", "method": "blocked_mean", "x": 5.0},
        {"subject": "sub-1", "recording": "b", "method": "aas", "x": 1.0},
        {"subject": "sub-2", "recording": "a", "method": "blocked_mean", "x": 2.0},
        {"subject": "sub-2", "recording": "a", "method": "aas", "x": 8.0},
    ]

    result = differences(rows, metric="x", comparator="aas")

    np.testing.assert_array_equal(result, [1.0, -6.0])


def test_bootstrap_interval_is_reproducible() -> None:
    script = Path(__file__).parents[2] / "tools" / "summarize_holdout.py"
    interval = runpy.run_path(str(script))["_bootstrap_median_interval"]
    values = np.array([-3.0, -2.0, -1.0, 0.0, 1.0])

    first = interval(values)
    second = interval(values)

    assert first == second
    assert first[0] <= np.median(values) <= first[1]


def test_holdout_summary_requires_every_current_metric(tmp_path: Path) -> None:
    script = Path(__file__).parents[2] / "tools" / "summarize_holdout.py"
    read_metrics = runpy.run_path(str(script))["_read_metrics"]
    path = tmp_path / "holdout_metrics.csv"
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=("subject", "recording", "method", "detector_status"),
        )
        writer.writeheader()
        writer.writerow(
            {
                "subject": "sub-1",
                "recording": "run1",
                "method": "blocked_mean",
                "detector_status": "ok",
            }
        )

    with np.testing.assert_raises_regex(ValueError, "missing metric columns"):
        read_metrics(path)
