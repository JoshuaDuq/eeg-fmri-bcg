"""Match FASTR, AAS, and BCGNet outputs for one recording."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..aas_batch import aas_output_vhdr
from ..discovery import iter_subjects
from .config import CompareConfig


@dataclass(frozen=True, slots=True)
class RecordingTriple:
    bids_id: str
    str_sub: str
    idx_run: int
    stem: str
    fastr_vhdr: Path
    aas_vhdr: Path | None
    bcgnet_mat: Path | None


def find_bcgnet_mat(bcgnet_root: Path, str_sub: str, idx_run: int) -> Path | None:
    name = f"{str_sub}_r0{idx_run}_bcgnet.mat"
    exact = [
        bcgnet_root / str_sub / name,
        bcgnet_root / str_sub / str_sub / name,
        bcgnet_root / name,
    ]
    for path in exact:
        if path.is_file():
            return path
    matches = sorted(bcgnet_root.glob(f"**/{name}"))
    return matches[0] if matches else None


def pair_recordings(config: CompareConfig) -> list[RecordingTriple]:
    triples: list[RecordingTriple] = []
    for bids_id, str_sub, vhdrs in iter_subjects(
        config.paths.fastr_root,
        include=config.include,
        exclude=config.exclude,
    ):
        for idx, src in enumerate(vhdrs, start=1):
            aas = aas_output_vhdr(config.paths.aas_root, bids_id, src)
            triples.append(
                RecordingTriple(
                    bids_id=bids_id,
                    str_sub=str_sub,
                    idx_run=idx,
                    stem=src.stem,
                    fastr_vhdr=src,
                    aas_vhdr=aas if aas.is_file() else None,
                    bcgnet_mat=find_bcgnet_mat(
                        config.paths.bcgnet_root, str_sub, idx
                    ),
                )
            )
    return triples
