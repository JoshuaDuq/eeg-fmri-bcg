"""CLI: bcgnet discover | run | aas | pca-obs | compare."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .compare.arms import AAS, PCA_OBS
from .config import ConfigurationError

#: Subcommands that generate one bounded arm with bcg_correction.
_COMPARATOR_COMMANDS = {"aas": AAS, "pca-obs": PCA_OBS}


def main(argv: list[str] | None = None) -> int:
    parser = _make_parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.command in {"discover", "run"}:
            return _run_bcgnet(arguments)
        if arguments.command in _COMPARATOR_COMMANDS:
            return _run_comparator(arguments)
        if arguments.command == "compare":
            return _run_compare(arguments)
    except ConfigurationError as error:
        print(f"{parser.prog}: {error}", file=sys.stderr)
        return 1
    return 0


def _run_bcgnet(arguments: argparse.Namespace) -> int:
    from .cohort import discover_subjects, run_cohort, run_count
    from .config import load_config

    config = load_config(arguments.config)
    if arguments.command == "discover":
        subjects = discover_subjects(config)
        payload = [
            {
                "bids_id": spec["bids_id"],
                "n_runs": run_count(spec),
                "n_recordings": len(spec["recordings"]),
                "recordings": [
                    {
                        "label": recording["label"],
                        "run": recording["run"],
                        "stem": recording["stem"],
                    }
                    for recording in spec["recordings"]
                ],
            }
            for spec in subjects
        ]
        print(json.dumps(payload, indent=2))
        return 0
    results = run_cohort(config, Path(arguments.config).expanduser().resolve())
    n_ok = sum(1 for result in results if result.get("status") == "ok")
    print(f"DONE ok={n_ok}/{len(results)}")
    return 0 if n_ok == len(results) else 1


def _run_comparator(arguments: argparse.Namespace) -> int:
    from . import correction_batch
    from .compare.config import load_compare_config

    arm = _COMPARATOR_COMMANDS[arguments.command]
    config = load_compare_config(arguments.config)
    rows = correction_batch.run_correction_batch(
        fastr_root=config.paths.fastr_root,
        output_root=config.paths.root_for(arm),
        arm=arm,
        settings=config.correction,
        include=config.include,
        exclude=config.exclude,
        workers=config.compute.workers,
    )
    n_ok = sum(1 for row in rows if row["status"] in {"ok", "skipped"})
    print(f"{arm.label} DONE ok_or_skipped={n_ok}/{len(rows)}")
    return 0 if n_ok == len(rows) else 1


def _run_compare(arguments: argparse.Namespace) -> int:
    from .compare.config import load_compare_config
    from .compare.pipeline import run_comparison

    rows = run_comparison(load_compare_config(arguments.config))
    print(f"COMPARE DONE recordings={len(rows)}")
    return 0


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bcgnet",
        description=(
            "Train BCGNet, run a bounded comparator arm, or compare them."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)
    discover = commands.add_parser(
        "discover",
        help="list FASTR subjects and runs without training",
    )
    run = commands.add_parser(
        "run",
        help="train one BCGNet GRU model per subject",
    )
    aas = commands.add_parser(
        "aas",
        help="run bundled AAS on every FASTR recording",
    )
    pca_obs = commands.add_parser(
        "pca-obs",
        help="run bundled PCA-OBS on every FASTR recording",
    )
    compare = commands.add_parser(
        "compare",
        help="plot Raw vs every corrected arm from existing folders",
    )
    for subparser in (discover, run, aas, pca_obs, compare):
        subparser.add_argument(
            "--config",
            "-c",
            type=Path,
            required=True,
            help="path to the YAML configuration",
        )
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
