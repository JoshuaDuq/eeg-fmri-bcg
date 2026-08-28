"""Plot Raw vs AAS vs BCGNet. Optionally generate a method first."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from ..aas_batch import run_aas_batch
from .config import CompareConfig
from .pairs import RecordingTriple, pair_recordings
from .plots import load_fastr, metrics_row, plot_epoch, plot_psd


def run_comparison(config: CompareConfig) -> list[dict]:
    if config.run.aas:
        print("Running AAS batch...")
        run_aas_batch(
            fastr_root=config.paths.fastr_root,
            aas_root=config.paths.aas_root,
            settings=config.aas,
            include=config.include,
            exclude=config.exclude,
        )
    if config.run.bcgnet:
        if config.bcgnet_config is None:
            raise RuntimeError("run.bcgnet is true but bcgnet_config is missing")
        from ..cohort import run_cohort
        from ..config import load_config

        print("Running BCGNet cohort...")
        run_cohort(load_config(config.bcgnet_config), config.bcgnet_config)

    triples = pair_recordings(config)
    rows: list[dict] = []
    fig_root = config.paths.output_root / "figures"
    for triple in triples:
        traces = _load_traces(triple)
        if "Raw" not in traces:
            continue
        if not any(name in traces for name in ("AAS", "BCGNet")):
            print(f"skip {triple.bids_id} {triple.stem}: no cleaned files")
            continue
        prefix = fig_root / triple.bids_id / f"run{triple.idx_run}"
        plot_psd(
            traces,
            title=f"Average PSD {triple.bids_id} run {triple.idx_run}",
            output=prefix.with_name(f"psd_run{triple.idx_run}_avg.png"),
            max_hz=config.plot.psd_max_hz,
        )
        plot_epoch(
            traces,
            channel=config.plot.channel,
            start=config.plot.epoch_start_seconds,
            duration=config.plot.epoch_seconds,
            title=f"{triple.bids_id} {config.plot.channel} run {triple.idx_run}",
            output=prefix.with_name(
                f"epoch_run{triple.idx_run}_{config.plot.channel}.png"
            ),
        )
        rows.append(
            metrics_row(
                triple,
                traces,
                max_hz=config.plot.psd_max_hz,
                window_seconds=config.aas.window_seconds,
            )
        )
        print(
            f"compared {triple.bids_id} run {triple.idx_run} "
            f"aas={triple.aas_vhdr is not None} "
            f"bcgnet={triple.bcgnet_vhdr is not None}"
        )
    _write_summary(config.paths.output_root, rows)
    return rows


def _load_traces(triple: RecordingTriple) -> dict:
    traces = {}
    try:
        raw = load_fastr(triple.fastr_vhdr)
    except Exception as error:
        print(f"failed to load FASTR {triple.fastr_vhdr}: {error}")
        return traces
    traces["Raw"] = raw
    if triple.aas_vhdr is not None:
        try:
            traces["AAS"] = load_fastr(triple.aas_vhdr)
        except Exception as error:
            print(f"failed to load AAS {triple.aas_vhdr}: {error}")
    if triple.bcgnet_vhdr is not None:
        try:
            traces["BCGNet"] = load_fastr(triple.bcgnet_vhdr)
        except Exception as error:
            print(f"failed to load BCGNet {triple.bcgnet_vhdr}: {error}")
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
