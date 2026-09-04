"""Plot Raw vs every corrected arm. Optionally generate an arm first."""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

from bcgstudy.correction_batch import run_correction_batch

matplotlib.use("Agg")

from .arms import BCGNET, CLEAN_ARMS, COMPARATOR_ARMS
from .comparative import save_comparative_report
from .config import CompareConfig
from .pairs import RecordingSet, pair_recordings
from .plots import (
    RAW_LABEL,
    load_fastr,
    measure_recording,
    metric_columns,
    metrics_row,
)


def _run_requested_arms(config: CompareConfig) -> None:
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
    if config.run.enabled(BCGNET):
        if config.bcgnet_config is None:
            raise RuntimeError("run.bcgnet is true but bcgnet_config is missing")
        from ..cohort import run_cohort
        from ..config import load_config

        print("Running BCGNet cohort...")
        run_cohort(load_config(config.bcgnet_config), config.bcgnet_config)


def _load_traces(recording: RecordingSet) -> dict:
    traces = {RAW_LABEL: load_fastr(recording.fastr_vhdr)}
    try:
        for arm in CLEAN_ARMS:
            if arm.key in recording.cleaned_vhdr:
                traces[arm.label] = load_fastr(recording.cleaned_vhdr[arm.key])
    except Exception:
        for raw in traces.values():
            raw.close()
        raise
    return traces


def paired_profiles(groups):
    """Intersect recording identities before any subject/cohort summary."""
    keyed = {}
    for key, items in groups.items():
        indexed = {(p.subject, p.label): p for p in items}
        if len(indexed) != len(items):
            raise ValueError(f"duplicate recording profile for {key}")
        keyed[key] = indexed
    common = (
        set.intersection(*(set(items) for items in keyed.values())) if keyed else set()
    )
    return {
        key: [items[identity] for identity in sorted(common)]
        for key, items in keyed.items()
    }


def _write_experiments(
    experiments_root: Path, profiles: dict, *, offered: dict[str, int]
) -> None:
    from bcg_correction.correction_report import (
        report_page_paths,
        save_topography_report,
    )

    active = [
        arm.key
        for arm in CLEAN_ARMS
        if any(arm.key in groups for groups in profiles.values())
    ]
    if not active:
        if offered or any(experiments_root.glob("cohort_*.*")):
            raise ValueError(
                "no evaluable profiles; reports not rebuilt, old files are stale"
            )
        return
    all_profiles = {
        key: [p for groups in profiles.values() for p in groups.get(key, [])]
        for key in active
    }
    cohort = paired_profiles(all_profiles)
    reference = next(iter(cohort.values()))
    if not reference:
        raise ValueError(
            "no paired cohort profiles; reports not rebuilt, old files are stale"
        )
    subject_reports = []
    for bids_id in sorted(offered):
        selected = {key: profiles.get(bids_id, {}).get(key, []) for key in active}
        paired = paired_profiles(selected)
        output = experiments_root / "subjects" / f"{bids_id}_comparative.png"
        if not any(paired.values()):
            stale = [output, output.with_suffix(".pdf")]
            stale.extend(
                path
                for page in report_page_paths(output).values()
                for path in (page, page.with_suffix(".pdf"))
            )
            if any(path.exists() for path in stale):
                raise ValueError(
                    f"no paired profiles for {bids_id}; reports not rebuilt, "
                    f"existing files are stale: {output}"
                )
            print(f"no paired profiles for {bids_id}; no subject page produced")
            continue
        subject_reports.append(
            (
                bids_id,
                paired,
                output,
                {key: offered[bids_id] - len(items) for key, items in selected.items()},
            )
        )
    for bids_id, paired, output, coverage in subject_reports:
        save_comparative_report(
            paired,
            title=f"{bids_id} — paired correction comparison",
            output=output,
            coverage=coverage,
        )
    subjects = len({p.subject for p in reference})
    title = f"Cohort — {len(reference)} paired recordings, {subjects} participants"
    coverage = {
        arm.key: sum(offered.values()) - len(all_profiles.get(arm.key, []))
        for arm in CLEAN_ARMS
    }
    save_comparative_report(
        cohort,
        title=title,
        output=experiments_root / "cohort_comparative.png",
        coverage=coverage,
    )
    save_topography_report(
        {arm.label: cohort[arm.key] for arm in CLEAN_ARMS if cohort.get(arm.key)},
        title=title,
        output=experiments_root / "cohort_topography.png",
    )
    print(
        f"paired cohort: {len(reference)}/{sum(offered.values())} recordings, "
        f"{subjects} participants"
    )
    print(f"experiments written to {experiments_root}")


def _write_summary(output_root: Path, rows: list[dict], evaluation) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / "compare_summary.json"
    csv_path = output_root / "compare_summary.csv"
    json_path.write_text(json.dumps(rows, indent=2, allow_nan=False))
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=metric_columns(evaluation))
        writer.writeheader()
        writer.writerows(rows)


def compare_existing_outputs(
    config: CompareConfig, *, plots_only: bool = False
) -> list[dict]:
    cache_path = config.paths.output_root / "compare_profiles.pkl"
    if plots_only:
        if not cache_path.exists():
            raise FileNotFoundError(
                f"No cached profiles found at {cache_path}. "
                "Run a full comparison first to generate the profile cache."
            )
        import pickle

        with cache_path.open("rb") as handle:
            cached = pickle.load(handle)
        _write_experiments(
            config.paths.experiments_root,
            cached["profiles"],
            offered=cached["offered"],
        )
        return cached.get("rows", [])

    recordings = pair_recordings(config)
    rows = []
    profiles = {}
    evaluation = config.correction.evaluation
    for recording in recordings:
        traces = _load_traces(recording)
        try:
            measured = measure_recording(recording, traces, evaluation)
            rows.append(
                metrics_row(
                    recording,
                    traces,
                    measured,
                    max_hz=config.plot.psd_max_hz,
                    evaluation=evaluation,
                )
            )
            for key, profile in measured.items():
                profiles.setdefault(recording.bids_id, {}).setdefault(key, []).append(
                    profile
                )
            print(
                f"compared {recording.bids_id} {recording.label} "
                f"profiles={len(measured)}"
            )
        finally:
            for raw in traces.values():
                raw.close()
    _write_summary(config.paths.output_root, rows, evaluation)
    offered = dict(Counter(recording.bids_id for recording in recordings))
    import pickle

    try:
        config.paths.output_root.mkdir(parents=True, exist_ok=True)
        with cache_path.open("wb") as handle:
            pickle.dump(
                {"profiles": profiles, "offered": offered, "rows": rows},
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
    except Exception as err:
        print(
            f"warning: could not cache comparison profiles: {err}",
            file=sys.stderr,
        )
    _write_experiments(
        config.paths.experiments_root,
        profiles,
        offered=offered,
    )
    return rows


def run_comparison(config: CompareConfig, *, plots_only: bool = False) -> list[dict]:
    if not plots_only:
        _run_requested_arms(config)
    return compare_existing_outputs(config, plots_only=plots_only)
