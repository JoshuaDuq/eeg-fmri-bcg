"""The ``bcg`` command: every pipeline in this project, under one name.

Subcommands are grouped by what they operate on rather than by which package
implements them, so a user picks a *method*, not a module:

    cohort      discover  aas  pca-obs  bcgnet  compare
    single      correct   detect  benchmark

No method is privileged. ``bcg aas`` and ``bcg bcgnet`` are siblings, which is
the whole point -- the previous layout ran AAS through a command named after the
neural network it is meant to be compared against.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from bcgnet.compare.arms import ARM_BY_COMMAND, COMPARATOR_ARMS


def _user_errors() -> tuple[type[Exception], ...]:
    from bcg_correction.bcg import BcgInputError
    from bcg_correction.bcg_benchmark import BenchmarkInputError
    from bcg_correction.brainvision import BrainVisionMarkerError
    from bcg_correction.brainvision_io import BrainVisionInputError
    from bcg_correction.cardiac import CardiacInputError
    from bcg_correction.cardiac_markers import CardiacMarkerError
    from bcg_correction.config import ConfigurationError as CorrectionConfigError
    from bcg_correction.metrics import MetricInputError
    from bcgnet.config import ConfigurationError

    return (
        BcgInputError,
        BenchmarkInputError,
        BrainVisionInputError,
        BrainVisionMarkerError,
        CardiacInputError,
        CardiacMarkerError,
        ConfigurationError,
        CorrectionConfigError,
        FileExistsError,
        MetricInputError,
    )


def _discover(arguments: argparse.Namespace) -> int:
    from bcgnet.cohort import discover_subjects, run_count
    from bcgnet.config import load_config

    subjects = discover_subjects(load_config(arguments.config))
    print(
        json.dumps(
            [
                {
                    "bids_id": spec["bids_id"],
                    "n_runs": run_count(spec),
                    "n_recordings": len(spec["recordings"]),
                    "recordings": [
                        {k: recording[k] for k in ("label", "run", "stem")}
                        for recording in spec["recordings"]
                    ],
                }
                for spec in subjects
            ],
            indent=2,
        )
    )
    return 0


def _bcgnet(arguments: argparse.Namespace) -> int:
    from bcgnet.cohort import run_cohort
    from bcgnet.config import load_config

    config_path = Path(arguments.config).expanduser().resolve()
    results = run_cohort(load_config(config_path), config_path)
    ok = sum(1 for result in results if result.get("status") == "ok")
    print(f"DONE ok={ok}/{len(results)}")
    return 0 if ok == len(results) else 1


def _comparator(arguments: argparse.Namespace) -> int:
    from bcgnet.compare.config import load_compare_config

    from .correction_batch import run_correction_batch

    arm = ARM_BY_COMMAND[arguments.command]
    config = load_compare_config(arguments.config)
    rows = run_correction_batch(
        fastr_root=config.paths.fastr_root,
        output_root=config.paths.root_for(arm),
        arm=arm,
        settings=config.correction,
        include=config.include,
        exclude=config.exclude,
        workers=config.compute.workers,
    )
    ok = sum(1 for row in rows if row["status"] in {"ok", "skipped"})
    print(f"{arm.label} DONE ok_or_skipped={ok}/{len(rows)}")
    return 0 if ok == len(rows) else 1


def _reports(arguments: argparse.Namespace) -> int:
    from bcgnet.compare.arms import COMPARATOR_ARMS
    from bcgnet.compare.config import load_compare_config
    from bcgnet.compare.pipeline import compare_existing_outputs

    from .correction_batch import rebuild_reports

    config = load_compare_config(arguments.config)
    total = 0
    for arm in COMPARATOR_ARMS:
        total += rebuild_reports(
            fastr_root=config.paths.fastr_root,
            output_root=config.paths.root_for(arm),
            arm=arm,
            settings=config.correction,
            include=config.include,
            exclude=config.exclude,
            workers=config.compute.workers,
        )["recordings"]
    compared = len(compare_existing_outputs(config))
    print(f"REPORTS DONE rebuilt={total} compared={compared}")
    return 0


def _compare(arguments: argparse.Namespace) -> int:
    from bcgnet.compare.config import load_compare_config
    from bcgnet.compare.pipeline import run_comparison

    plots_only = getattr(arguments, "plots_only", False)
    rows = run_comparison(
        load_compare_config(arguments.config), plots_only=plots_only
    )
    if plots_only:
        print("COMPARE DONE plots_only=True")
    else:
        print(f"COMPARE DONE recordings={len(rows)}")
    return 0


def _single(arguments: argparse.Namespace) -> int:
    from bcg_correction.bcg_benchmark import run_bcg_benchmark
    from bcg_correction.bcg_config import (
        load_benchmark_config,
        load_correction_config,
        load_detection_config,
    )
    from bcg_correction.bcg_pipeline import run_bcg_correction
    from bcg_correction.cardiac_pipeline import run_cardiac_detection

    load, run = {
        "correct": (load_correction_config, run_bcg_correction),
        "detect": (load_detection_config, run_cardiac_detection),
        "benchmark": (load_benchmark_config, run_bcg_benchmark),
    }[arguments.command]
    print(json.dumps(asdict(run(load(arguments.config))), indent=2, default=str))
    return 0


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bcg",
        description=(
            "Ballistocardiogram correction for EEG-fMRI. Three methods on equal "
            "footing -- AAS, PCA-OBS, BCGNet -- plus a "
            "comparison between them."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    cohort = (
        (
            "discover",
            "list subjects and recordings without correcting anything",
            "config.yaml",
        ),
        (
            "aas",
            "correct every recording with average artifact subtraction",
            "compare.yaml",
        ),
        (
            "pca-obs",
            "correct every recording with the optimal basis set",
            "compare.yaml",
        ),
        (
            "bcgnet",
            "train one BCGNet GRU per subject and correct with it",
            "config.yaml",
        ),
        ("compare", "score every arm that ran against the others", "compare.yaml"),
        (
            "reports",
            "rebuild correction and comparison reports from outputs on disk",
            "compare.yaml",
        ),
    )
    single = (
        ("correct", "correct one recording", "examples/bcg_correction.yml"),
        (
            "detect",
            "derive ECG R markers for one recording",
            "examples/cardiac_detection.yml",
        ),
        (
            "benchmark",
            "compare bounded corrections on paired recordings",
            "examples/bcg_benchmark.yml",
        ),
    )
    for name, help_text, example in cohort + single:
        sub = commands.add_parser(name, help=help_text)
        sub.add_argument(
            "--config",
            "-c",
            type=Path,
            required=True,
            help=f"path to the YAML configuration, e.g. {example}",
        )
        if name == "compare":
            sub.add_argument(
                "--plots-only",
                action="store_true",
                help=(
                    "rebuild comparative plots from cached profiles without "
                    "re-reading EEG"
                ),
            )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _make_parser()
    arguments = parser.parse_args(argv)
    handlers = {
        "discover": _discover,
        "bcgnet": _bcgnet,
        "compare": _compare,
        "reports": _reports,
        "correct": _single,
        "detect": _single,
        "benchmark": _single,
    }
    handlers.update({arm.command: _comparator for arm in COMPARATOR_ARMS})
    handler = handlers[arguments.command]
    try:
        return handler(arguments)
    except _user_errors() as error:
        print(f"bcg {arguments.command}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
