"""Plot Raw vs every corrected arm. Optionally generate an arm first."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from ..correction_batch import run_correction_batch
from .arms import CLEAN_ARMS, COMPARATOR_ARMS
from .config import CompareConfig
from .pairs import RecordingSet, pair_recordings
from .plots import RAW_LABEL, load_fastr, metrics_row, plot_epoch, plot_psd


def run_comparison(config: CompareConfig) -> list[dict]:
    for arm in COMPARATOR_ARMS:
        if not config.run.enabled(arm):
            continue
        print(f"Running {arm.label} batch...")
        run_correction_batch(
            fastr_root=config.paths.fastr_root,
            output_root=config.paths.root_for(arm),
            arm=arm,
            settings=config.correction,
            include=config.include,
            exclude=config.exclude,
            workers=config.compute.workers,
        )
    if config.run.bcgnet:
        if config.bcgnet_config is None:
            raise RuntimeError("run.bcgnet is true but bcgnet_config is missing")
        from ..cohort import run_cohort
        from ..config import load_config

        print("Running BCGNet cohort...")
        run_cohort(load_config(config.bcgnet_config), config.bcgnet_config)

    recordings = pair_recordings(config)
    rows: list[dict] = []
    fig_root = config.paths.output_root / "figures"
    for recording in recordings:
        traces = _load_traces(recording)
        if RAW_LABEL not in traces:
            continue
        present = [arm.label for arm in CLEAN_ARMS if arm.label in traces]
        if not present:
            print(f"skip {recording.bids_id} {recording.stem}: no cleaned files")
            continue
        label = recording.label
        prefix = fig_root / recording.bids_id / label
        plot_psd(
            traces,
            title=f"Average PSD {recording.bids_id} {label}",
            output=prefix.with_name(f"psd_{label}_avg.png"),
            max_hz=config.plot.psd_max_hz,
        )
        plot_epoch(
            traces,
            channel=config.plot.channel,
            start=config.plot.epoch_start_seconds,
            duration=config.plot.epoch_seconds,
            title=(
                f"{recording.bids_id} {config.plot.channel} {label}"
            ),
            output=prefix.with_name(
                f"epoch_{label}_{config.plot.channel}.png"
            ),
        )
        rows.append(
            metrics_row(
                recording,
                traces,
                max_hz=config.plot.psd_max_hz,
                window_seconds=config.correction.window_seconds,
            )
        )
        print(
            f"compared {recording.bids_id} {recording.label} "
            f"arms={'+'.join(present)}"
        )
    _write_summary(config.paths.output_root, rows)
    return rows


def _load_traces(recording: RecordingSet) -> dict:
    traces = {}
    try:
        traces[RAW_LABEL] = load_fastr(recording.fastr_vhdr)
    except Exception as error:
        print(f"failed to load FASTR {recording.fastr_vhdr}: {error}")
        return traces
    for arm in CLEAN_ARMS:
        vhdr = recording.cleaned_vhdr.get(arm.key)
        if vhdr is None:
            continue
        try:
            traces[arm.label] = load_fastr(vhdr)
        except Exception as error:
            print(f"failed to load {arm.label} {vhdr}: {error}")
    return traces


def _write_summary(output_root: Path, rows: list[dict]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / "compare_summary.json"
    csv_path = output_root / "compare_summary.csv"
    json_path.write_text(json.dumps(rows, indent=2))
    if not rows:
        return
    keys = list(rows[0].keys())
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
