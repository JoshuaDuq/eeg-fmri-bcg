"""Batch a bounded correction arm over FASTR recordings via bcg_correction."""

from __future__ import annotations

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


def run_correction_batch(
    *,
    fastr_root: Path,
    output_root: Path,
    arm: Arm,
    settings: CorrectionSettings,
    include: tuple[str, ...] = (),
    exclude: tuple[str, ...] = (),
) -> list[dict]:
    results: list[dict] = []
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
            if output.exists() and not settings.overwrite:
                row["status"] = "skipped"
                results.append(row)
                print(f"skip {arm.label} (exists) {bids_id} {src.name}")
                continue
            config = CorrectionRunConfig(
                input_vhdr=src,
                output_vhdr=output,
                detector=settings.detector,
                method=arm.key,
                window_seconds=settings.window_seconds,
                ecg_to_bcg_delay_seconds=settings.ecg_to_bcg_delay_seconds,
                aas_neighbor_count=settings.aas_neighbor_count,
                pca_obs_components=settings.pca_obs_components,
                maximum_residual_ratio=settings.maximum_residual_ratio,
            )
            try:
                summary = run_bcg_correction(config)
                row["status"] = "ok"
                row["marker_count"] = summary.marker_count
            except FileExistsError:
                row["status"] = "skipped"
            except Exception as error:
                row["status"] = "error"
                row["error"] = f"{type(error).__name__}: {error}"
                print(f"{arm.label} failed {bids_id} {src.name}: {row['error']}")
            else:
                print(f"{arm.label} ok {bids_id} {src.name}")
            results.append(row)
    return results
