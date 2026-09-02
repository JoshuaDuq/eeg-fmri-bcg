import runpy
from pathlib import Path

import yaml


def test_holdout_config_is_strict_and_resolves_paths(tmp_path) -> None:
    script = Path(__file__).parents[2] / "tools" / "validate_blocked_mean.py"
    load_config = runpy.run_path(str(script))["_load_config"]
    path = tmp_path / "holdout.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "compare_config": "compare.yaml",
                "output_root": "results",
                "recordings": [
                    {"subject": "sub-0001", "label": "BaselineEEG"},
                    {"subject": "sub-0003", "label": "run3"},
                ],
                "cross_fit_fold_count": 10,
            }
        )
    )

    config = load_config(path)

    assert config.compare_config == tmp_path / "compare.yaml"
    assert config.output_root == tmp_path / "results"
    assert [(item.subject, item.label) for item in config.recordings] == [
        ("sub-0001", "BaselineEEG"),
        ("sub-0003", "run3"),
    ]
    assert config.cross_fit_fold_count == 10
