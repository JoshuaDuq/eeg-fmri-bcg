import runpy
from pathlib import Path

import numpy as np
import yaml


def test_robustness_config_rejects_duplicate_fold_counts(tmp_path) -> None:
    script = (
        Path(__file__).parents[2]
        / "tools"
        / "experiment_blocked_mean_robustness.py"
    )
    load_config = runpy.run_path(str(script))["_load_config"]
    path = tmp_path / "robustness.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "compare_config": "compare.yaml",
                "holdout_metrics": "holdout.csv",
                "output_root": "results",
                "cross_fit_fold_counts": [2, 5, 5],
                "injection_amplitude_uv": 5.0,
                "event_count": 120,
                "seed": 7,
            }
        )
    )

    with np.testing.assert_raises_regex(ValueError, "unique"):
        load_config(path)


def test_known_injections_are_deterministic_and_nonzero() -> None:
    script = (
        Path(__file__).parents[2]
        / "tools"
        / "experiment_blocked_mean_robustness.py"
    )
    make_injections = runpy.run_path(str(script))["_make_injections"]

    first = make_injections(
        channel_count=3,
        sample_count=2_000,
        sampling_rate=1_000.0,
        amplitude_uv=5.0,
        event_count=12,
        seed=42,
    )
    second = make_injections(
        channel_count=3,
        sample_count=2_000,
        sampling_rate=1_000.0,
        amplitude_uv=5.0,
        event_count=12,
        seed=42,
    )

    assert tuple(first) == ("tone_10hz", "sparse_events")
    for name in first:
        np.testing.assert_array_equal(first[name], second[name])
        assert first[name].shape == (3, 2_000)
        assert np.linalg.norm(first[name]) > 0.0
