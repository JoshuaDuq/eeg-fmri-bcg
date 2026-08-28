"""CLI: bcgnet discover | run | aas | compare."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import ConfigurationError


def main(argv: list[str] | None = None) -> int:
    parser = _make_parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.command in {"discover", "run"}:
            return _run_bcgnet(arguments)
        if arguments.command == "aas":
            return _run_aas(arguments)
        if arguments.command == "compare":
            return _run_compare(arguments)
    except ConfigurationError as error:
        print(f"{parser.prog}: {error}", file=sys.stderr)
        return 1
    return 0


def _run_bcgnet(arguments: argparse.Namespace) -> int:
    from .cohort import discover_subjects, run_cohort
    from .config import load_config

    config = load_config(arguments.config)
    if arguments.command == "discover":
        subjects = discover_subjects(config)
        payload = [
            {
                "bids_id": spec["bids_id"],
                "n_runs": len(spec["runs"]),
                "runs": [run["stem"] for run in spec["runs"]],
            }
            for spec in subjects
        ]
        print(json.dumps(payload, indent=2))
        return 0
    results = run_cohort(config, Path(arguments.config).expanduser().resolve())
    n_ok = sum(1 for result in results if result.get("status") == "ok")
    print(f"DONE ok={n_ok}/{len(results)}")
    return 0 if n_ok == len(results) else 1


def _run_aas(arguments: argparse.Namespace) -> int:
    from .aas_batch import run_aas_batch
    from .compare.config import load_compare_config

    config = load_compare_config(arguments.config)
    rows = run_aas_batch(
        fastr_root=config.paths.fastr_root,
        aas_root=config.paths.aas_root,
        settings=config.aas,
        include=config.include,
        exclude=config.exclude,
    )
    n_ok = sum(1 for row in rows if row["status"] in {"ok", "skipped"})
    print(f"AAS DONE ok_or_skipped={n_ok}/{len(rows)}")
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
        description="Train BCGNet, run AAS, or compare Raw vs AAS vs BCGNet.",
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
    compare = commands.add_parser(
        "compare",
        help="plot Raw vs AAS vs BCGNet from existing folders",
    )
    for subparser in (discover, run, aas, compare):
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
