"""Orchestrate a bounded correction arm over FASTR recordings."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from itertools import repeat
from pathlib import Path

from bcg_correction.bcg_config import CorrectionRunConfig, DetectorConfig
from bcg_correction.bcg_pipeline import run_bcg_correction
from bcg_correction.evaluation import EvaluationSettings
from bcg_correction.provenance import load_correction_provenance
from bcgnet.compare.arms import Arm

from .discovery import iter_subjects


@dataclass(frozen=True, slots=True)
class CorrectionSettings:
    """Settings shared by every bounded comparator arm.

    The bounded arms run the same independent R detector over the same window, so the
    only per-arm inputs are the method itself and its own component/neighbour
    count. Keeping one block means the arms cannot silently drift apart.
    """

    window_seconds: tuple[float, float]
    ecg_to_bcg_delay_seconds: float
    aas_neighbor_count: int
    pca_obs_components: int
    evaluation: EvaluationSettings
    maximum_gap_fraction: float
    overwrite: bool
    detector: DetectorConfig


def correction_output_vhdr(
    root: Path, bids_id: str, fastr_vhdr: Path, *, arm: Arm
) -> Path:
    return root / bids_id / f"{fastr_vhdr.stem}_{arm.suffix}.vhdr"


def _correct_one(config: CorrectionRunConfig) -> dict:
    try:
        summary = run_bcg_correction(config)
    except FileExistsError:
        return {"status": "skipped"}
    except Exception as error:
        return {"status": "error", "error": f"{type(error).__name__}: {error}"}
    return {"status": "ok", "marker_count": summary.marker_count}


def _map_corrections(configs: list[CorrectionRunConfig], workers: int) -> list[dict]:
    if workers <= 1 or len(configs) <= 1:
        return [_correct_one(config) for config in configs]
    with ProcessPoolExecutor(max_workers=min(workers, len(configs))) as pool:
        return list(pool.map(_correct_one, configs))


def _report(arm: Arm, row: dict) -> None:
    name = Path(row["input"]).name
    if row["status"] == "error":
        print(f"{arm.label} failed {row['bids_id']} {name}: {row['error']}")
    elif row["status"] == "ok":
        print(f"{arm.label} ok {row['bids_id']} {name}")


def _rebuild_one(
    job: tuple[Path, Path], arm: Arm, settings: CorrectionSettings
) -> bool:
    import mne

    from bcg_correction.correction_report import (
        compute_correction_profile,
        save_correction_report,
        write_profile,
    )

    source, corrected = job
    provenance = load_correction_provenance(corrected)
    if provenance is None:
        raise FileNotFoundError(corrected.with_suffix(".bcg.json"))
    raw = mne.io.read_raw_brainvision(source, preload=True, verbose="ERROR")
    clean = mne.io.read_raw_brainvision(corrected, preload=True, verbose="ERROR")
    try:
        if (
            raw.n_times != clean.n_times
            or raw.ch_names != clean.ch_names
            or raw.info["sfreq"] != clean.info["sfreq"]
            or "ECG" not in raw.ch_names
        ):
            raise ValueError(f"unaligned report input: {corrected}")
        profile = compute_correction_profile(
            raw.get_data(),
            clean.get_data(),
            tuple(raw.ch_names),
            ecg_channel_index=raw.ch_names.index("ECG"),
            peak_samples=provenance.peak_samples,
            sampling_rate_hz=float(raw.info["sfreq"]),
            delay_seconds=provenance.delay_seconds,
            window_seconds=provenance.window_seconds,
            gap_fraction=provenance.gap_fraction,
            method=arm.key,
            label=source.stem,
            subject=corrected.parent.name,
            evaluation=settings.evaluation,
        )
    finally:
        raw.close()
        clean.close()
    if profile is None:
        return False
    stem = corrected.with_suffix("")
    save_correction_report(
        profile,
        title=f"{corrected.stem}  \u2014  {arm.label} correction report",
        output=Path(f"{stem}_correction_report.png"),
    )
    write_profile(profile, Path(f"{stem}_profile.npz"))
    return True


def _map_rebuilds(jobs, arm, settings, workers: int) -> list[bool]:
    if workers <= 1:
        return [_rebuild_one(job, arm, settings) for job in jobs]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_rebuild_one, jobs, repeat(arm), repeat(settings)))


def write_aggregate_reports(output_root: Path, arm: Arm) -> dict[str, int]:
    """Grand-average the per-recording profiles into subject and cohort pages."""
    from bcg_correction.correction_report import (
        read_profile,
        save_aggregate_report,
        save_topography_report,
    )

    by_subject: dict[str, list] = {}
    for stored in sorted(output_root.glob("*/*_profile.npz")):
        if stored.name.startswith("._"):
            continue
        profile = read_profile(stored)
        if profile.method != arm.key:
            raise ValueError(f"unexpected method profile: {stored}")
        profile = replace(profile, subject=stored.parent.name)
        by_subject.setdefault(stored.parent.name, []).append(profile)
    if not by_subject:
        return {"subjects": 0, "recordings": 0}
    report_root = output_root / "reports"
    everything = []
    for bids_id, profiles in sorted(by_subject.items()):
        everything.extend(profiles)
        save_aggregate_report(
            profiles,
            title=(
                f"{bids_id}  \u2014  {arm.label} subject report, "
                f"{len(profiles)} recordings"
            ),
            output=report_root / f"{bids_id}_{arm.suffix}_report.png",
        )
    save_aggregate_report(
        everything,
        title=(
            f"{arm.label} cohort report  \u2014  {len(everything)} recordings, "
            f"{len(by_subject)} subjects"
        ),
        output=report_root / f"cohort_{arm.suffix}_report.png",
    )
    save_topography_report(
        {arm.label: everything},
        title=(
            f"{arm.label}  \u2014  where it acts, {len(everything)} recordings, "
            f"{len(by_subject)} subjects"
        ),
        output=report_root / f"cohort_{arm.suffix}_topography.png",
    )
    print(
        f"{arm.label} reports: {len(by_subject)} subject pages + cohort page "
        f"in {report_root}"
    )
    return {"subjects": len(by_subject), "recordings": len(everything)}


def run_correction_batch(
    *,
    fastr_root: Path,
    output_root: Path,
    arm: Arm,
    settings: CorrectionSettings,
    include: tuple[str, ...] = (),
    exclude: tuple[str, ...] = (),
    workers: int = 1,
) -> list[dict]:
    results: list[dict] = []
    pending: list[tuple[dict, CorrectionRunConfig]] = []
    for bids_id, _str_sub, recordings in iter_subjects(
        fastr_root, include=include, exclude=exclude
    ):
        for recording in recordings:
            src = recording.path
            output = correction_output_vhdr(output_root, bids_id, src, arm=arm)
            row = {
                "bids_id": bids_id,
                "method": arm.key,
                "input": str(src),
                "output": str(output),
            }
            results.append(row)
            if output.exists() and not settings.overwrite:
                row["status"] = "skipped"
                print(f"skip {arm.label} (exists) {bids_id} {src.name}")
                continue
            pending.append(
                (
                    row,
                    CorrectionRunConfig(
                        input_vhdr=src,
                        output_vhdr=output,
                        detector=settings.detector,
                        method=arm.key,
                        window_seconds=settings.window_seconds,
                        ecg_to_bcg_delay_seconds=(settings.ecg_to_bcg_delay_seconds),
                        aas_neighbor_count=settings.aas_neighbor_count,
                        pca_obs_components=settings.pca_obs_components,
                        evaluation=settings.evaluation,
                        maximum_gap_fraction=settings.maximum_gap_fraction,
                    ),
                )
            )
    outcomes = _map_corrections([config for _row, config in pending], workers)
    for (row, _config), outcome in zip(pending, outcomes, strict=True):
        row.update(outcome)
        _report(arm, row)
    write_aggregate_reports(output_root, arm)
    return results


def rebuild_reports(
    *,
    fastr_root: Path,
    output_root: Path,
    arm: Arm,
    settings: CorrectionSettings,
    include: tuple[str, ...] = (),
    exclude: tuple[str, ...] = (),
    workers: int = 1,
) -> dict[str, int]:
    jobs: list[tuple[Path, Path]] = []
    for bids_id, _str_sub, recordings in iter_subjects(
        fastr_root, include=include, exclude=exclude
    ):
        for recording in recordings:
            corrected = correction_output_vhdr(
                output_root, bids_id, recording.path, arm=arm
            )
            if corrected.with_suffix(".bcg.json").is_file():
                jobs.append((recording.path, corrected))
    if not jobs:
        print(f"{arm.label}: nothing to rebuild under {output_root}")
        return {"recordings": 0}
    print(f"{arm.label}: rebuilding {len(jobs)} report(s) from existing output")
    rebuilt = _map_rebuilds(jobs, arm, settings, workers)
    ok = sum(1 for done in rebuilt if done)
    write_aggregate_reports(output_root, arm)
    return {"recordings": ok}
