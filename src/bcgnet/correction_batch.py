"""Batch a bounded correction arm over FASTR recordings via bcg_correction."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from bcg_correction.bcg_config import CorrectionRunConfig, DetectorConfig
from bcg_correction.bcg_pipeline import run_bcg_correction

from .compare.arms import Arm
from .discovery import iter_subjects


@dataclass(frozen=True, slots=True)
class CorrectionSettings:
    """Settings shared by every bounded comparator arm.

    Both arms run the same independent R detector over the same window, so the
    only per-arm inputs are the method itself and its own component/neighbour
    count. Keeping one block means the arms cannot silently drift apart.
    """

    window_seconds: tuple[float, float]
    ecg_to_bcg_delay_seconds: float
    aas_neighbor_count: int
    pca_obs_components: int
    maximum_residual_ratio: float
    overwrite: bool
    detector: DetectorConfig


def correction_output_vhdr(
    root: Path, bids_id: str, fastr_vhdr: Path, *, arm: Arm
) -> Path:
    return root / bids_id / f"{fastr_vhdr.stem}_{arm.suffix}.vhdr"


def _correct_one(config: CorrectionRunConfig) -> dict:
    """Run one bounded correction, reporting failure as data rather than raising.

    This is what crosses into a worker process, so the outcome has to survive
    pickling: an arbitrary exception instance may not, a plain dict always does.
    """
    try:
        summary = run_bcg_correction(config)
    except FileExistsError:
        return {"status": "skipped"}
    except Exception as error:
        return {"status": "error", "error": f"{type(error).__name__}: {error}"}
    return {"status": "ok", "marker_count": summary.marker_count}


def _map_corrections(
    configs: list[CorrectionRunConfig], workers: int
) -> list[dict]:
    """Correct every config, in worker processes when there is more than one.

    Recordings are independent, so the only thing the pool has to preserve is
    order -- ``map`` does, which keeps the batch's report deterministic however
    many workers ran it.
    """
    if workers <= 1 or len(configs) <= 1:
        return [_correct_one(config) for config in configs]
    with ProcessPoolExecutor(max_workers=min(workers, len(configs))) as pool:
        return list(pool.map(_correct_one, configs))


def _report(arm: Arm, row: dict) -> None:
    """Print one finished recording. A skip is silent, as it always was."""
    name = Path(row["input"]).name
    if row["status"] == "error":
        print(f"{arm.label} failed {row['bids_id']} {name}: {row['error']}")
    elif row["status"] == "ok":
        print(f"{arm.label} ok {row['bids_id']} {name}")


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
    """Correct every discovered recording with ``arm``.

    :param workers: how many recordings to correct at once. One keeps the batch
        in this process; more spreads independent recordings over that many
        worker processes.
    """
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
                        ecg_to_bcg_delay_seconds=(
                            settings.ecg_to_bcg_delay_seconds
                        ),
                        aas_neighbor_count=settings.aas_neighbor_count,
                        pca_obs_components=settings.pca_obs_components,
                        maximum_residual_ratio=settings.maximum_residual_ratio,
                    ),
                )
            )
    outcomes = _map_corrections([config for _row, config in pending], workers)
    for (row, _config), outcome in zip(pending, outcomes, strict=True):
        row.update(outcome)
        _report(arm, row)
    return results
