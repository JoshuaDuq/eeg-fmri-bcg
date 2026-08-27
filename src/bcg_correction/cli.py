"""Command-line entry points for the AAS/PCA-OBS BCG pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from .bcg import BcgInputError
from .bcg_benchmark import BenchmarkInputError, run_bcg_benchmark
from .bcg_config import (
    load_benchmark_config,
    load_correction_config,
    load_detection_config,
)
from .bcg_pipeline import run_bcg_correction
from .brainvision import BrainVisionMarkerError
from .brainvision_io import BrainVisionInputError
from .cardiac import CardiacInputError
from .cardiac_markers import CardiacMarkerError
from .cardiac_pipeline import run_cardiac_detection
from .config import ConfigurationError
from .metrics import MetricInputError


def main(argv: list[str] | None = None) -> int:
    parser = _make_parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "detect-cardiac":
            _print_summary(run_cardiac_detection(load_detection_config(arguments.config)))
        elif arguments.command == "correct-bcg":
            _print_summary(run_bcg_correction(load_correction_config(arguments.config)))
        elif arguments.command == "benchmark-bcg":
            _print_summary(run_bcg_benchmark(load_benchmark_config(arguments.config)))
    except (
        BcgInputError,
        BenchmarkInputError,
        BrainVisionInputError,
        BrainVisionMarkerError,
        CardiacInputError,
        CardiacMarkerError,
        ConfigurationError,
        FileExistsError,
        MetricInputError,
    ) as error:
        print(f"{parser.prog}: {error}", file=sys.stderr)
        return 1
    return 0


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bcg-correct",
        description=(
            "Detect independent ECG R markers and subtract ballistocardiogram "
            "artifact from FASTR-corrected EEG."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)
    detect_cardiac = commands.add_parser(
        "detect-cardiac",
        help="derive independent ECG R markers from a FASTR-only recording",
    )
    detect_cardiac.add_argument("--config", type=Path, required=True)
    correct_bcg = commands.add_parser(
        "correct-bcg",
        help="detect independent R markers and subtract BCG from EEG channels",
    )
    correct_bcg.add_argument("--config", type=Path, required=True)
    benchmark_bcg = commands.add_parser(
        "benchmark-bcg",
        help="compare bounded BCG corrections on paired recordings",
    )
    benchmark_bcg.add_argument("--config", type=Path, required=True)
    return parser


def _print_summary(summary: object) -> None:
    print(json.dumps(asdict(summary), indent=2, default=str))


if __name__ == "__main__":
    raise SystemExit(main())
