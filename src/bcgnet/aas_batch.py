"""Batch AAS/PCA-OBS correction using the bundled bcg_correction library."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from bcg_correction.bcg_config import CorrectionRunConfig, DetectorConfig
from bcg_correction.bcg_pipeline import run_bcg_correction

from .discovery import iter_subjects


@dataclass(frozen=True, slots=True)
class AasSettings:
    method: str
    window_seconds: tuple[float, float]
    ecg_to_bcg_delay_seconds: float
    aas_neighbor_count: int
    pca_obs_components: int
    maximum_residual_ratio: float
    overwrite: bool
    detector: DetectorConfig


def aas_output_vhdr(aas_root: Path, bids_id: str, fastr_vhdr: Path) -> Path:
    return aas_root / bids_id / f"{fastr_vhdr.stem}_bcg.vhdr"


def run_aas_batch(
    *,
    fastr_root: Path,
    aas_root: Path,
    settings: AasSettings,
    include: tuple[str, ...] = (),
    exclude: tuple[str, ...] = (),
) -> list[dict]:
    results: list[dict] = []
    for bids_id, _str_sub, vhdrs in iter_subjects(
        fastr_root, include=include, exclude=exclude
    ):
        for src in vhdrs:
            output = aas_output_vhdr(aas_root, bids_id, src)
            row = {
                "bids_id": bids_id,
                "input": str(src),
                "output": str(output),
            }
            if output.exists() and not settings.overwrite:
                row["status"] = "skipped"
                results.append(row)
                print(f"skip AAS (exists) {bids_id} {src.name}")
                continue
            config = CorrectionRunConfig(
                input_vhdr=src,
                output_vhdr=output,
                detector=settings.detector,
                method=settings.method,
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
                print(f"AAS failed {bids_id} {src.name}: {row['error']}")
            else:
                print(f"AAS ok {bids_id} {src.name}")
            results.append(row)
    return results
