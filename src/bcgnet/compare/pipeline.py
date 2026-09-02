"""Plot Raw vs every corrected arm. Optionally generate an arm first."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

from bcgstudy.correction_batch import run_correction_batch

matplotlib.use("Agg")

from .arms import BCGNET, CLEAN_ARMS, COMPARATOR_ARMS
from .comparative import save_comparative_report
from .config import CompareConfig
from .pairs import RecordingSet, pair_recordings
from .plots import (
    METRIC_COLUMNS,
    RAW_LABEL,
    detector_provenance,
    load_fastr,
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


def _collect_profiles(recording, traces, profiles) -> None:
    from bcg_correction.correction_report import compute_correction_profile

    provenance = detector_provenance(recording)
    if provenance is None:
        return
    raw = traces[RAW_LABEL]
    if "ECG" not in raw.ch_names:
        return
    for arm in CLEAN_ARMS:
        cleaned = traces.get(arm.label)
        if cleaned is None or cleaned.n_times != raw.n_times:
            continue
        profile = compute_correction_profile(
            raw.get_data(),
            cleaned.get_data(),
            tuple(raw.ch_names),
            ecg_channel_index=raw.ch_names.index("ECG"),
            peak_samples=provenance.peak_samples,
            sampling_rate_hz=float(raw.info["sfreq"]),
            delay_seconds=provenance.delay_seconds,
            window_seconds=provenance.window_seconds,
            gap_fraction=provenance.gap_fraction,
            method=arm.key,
            label=recording.label,
        )
        if profile is not None:
            profiles.setdefault(recording.bids_id, {}).setdefault(
                arm.key, []
            ).append(profile)


def _write_experiments(
    experiments_root: Path, profiles: dict, *, offered: int
) -> None:
    if not profiles:
        return
    keyed: dict[str, dict[tuple[str, str], object]] = {}
    for bids_id, by_arm in sorted(profiles.items()):
        save_comparative_report(
            by_arm,
            title=f"{bids_id}  \u2014  correction methods compared",
            output=experiments_root / "subjects" / f"{bids_id}_comparative.png",
        )
        for key, items in by_arm.items():
            for profile in items:
                keyed.setdefault(key, {})[(bids_id, profile.label)] = profile

    produced = {key: len(items) for key, items in keyed.items()}
    common: set[tuple[str, str]] = set.intersection(
        *(set(items) for items in keyed.values())
    ) if keyed else set()
    cohort = {
        key: [items[recording] for recording in sorted(common)]
        for key, items in keyed.items()
    }
    dropped = {key: n - len(common) for key, n in produced.items()}
    for key, count in sorted(dropped.items()):
        if count:
            print(f"cohort pairing: {key} drops {count} unpaired recording(s)")
    save_comparative_report(
        cohort,
        title=(
            f"Cohort  \u2014  correction methods compared, "
            f"{len(common)} paired recordings, {len(profiles)} subjects"
        ),
        output=experiments_root / "cohort_comparative.png",
        coverage={key: offered - n for key, n in produced.items()},
    )
    from bcg_correction.correction_report import save_topography_report

    from .arms import CLEAN_ARMS as _ARMS

    save_topography_report(
        {arm.label: cohort[arm.key] for arm in _ARMS if cohort.get(arm.key)},
        title=(
            f"Cohort  \u2014  where each method acts, "
            f"{len(common)} paired recordings, {len(profiles)} subjects"
        ),
        output=experiments_root / "cohort_topography.png",
    )
    print(f"experiments written to {experiments_root}")


def _write_summary(output_root: Path, rows: list[dict]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / "compare_summary.json"
    csv_path = output_root / "compare_summary.csv"
    json_path.write_text(json.dumps(rows, indent=2))
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def compare_existing_outputs(config: CompareConfig) -> list[dict]:
    recordings = pair_recordings(config)
    rows: list[dict] = []
    profiles: dict[str, dict[str, list]] = {}
    for recording in recordings:
        traces = _load_traces(recording)
        if RAW_LABEL not in traces:
            continue
        present = [arm.label for arm in CLEAN_ARMS if arm.label in traces]
        if not present:
            print(f"skip {recording.bids_id} {recording.stem}: no cleaned files")
            continue
        rows.append(
            metrics_row(
                recording,
                traces,
                max_hz=config.plot.psd_max_hz,
            )
        )
        _collect_profiles(recording, traces, profiles)
        print(
            f"compared {recording.bids_id} {recording.label} "
            f"arms={'+'.join(present)}"
        )
    _write_summary(config.paths.output_root, rows)
    _write_experiments(
        config.paths.experiments_root,
        profiles,
        offered=len(recordings),
    )
    return rows


def run_comparison(config: CompareConfig) -> list[dict]:
    _run_requested_arms(config)
    return compare_existing_outputs(config)
