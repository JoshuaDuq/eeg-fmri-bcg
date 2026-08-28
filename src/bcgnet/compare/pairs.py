"""Match FASTR, AAS, and BCGNet outputs for one recording."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..aas_batch import aas_output_vhdr
from ..discovery import iter_subjects
from ..export import bcgnet_output_vhdr
from .config import CompareConfig


@dataclass(frozen=True, slots=True)
class RecordingTriple:
    bids_id: str
    str_sub: str
    idx_run: int
    stem: str
    fastr_vhdr: Path
    aas_vhdr: Path | None
    bcgnet_vhdr: Path | None


def pair_recordings(config: CompareConfig) -> list[RecordingTriple]:
    triples: list[RecordingTriple] = []
    for bids_id, str_sub, vhdrs in iter_subjects(
        config.paths.fastr_root,
        include=config.include,
        exclude=config.exclude,
    ):
        for idx, src in enumerate(vhdrs, start=1):
            aas = aas_output_vhdr(config.paths.aas_root, bids_id, src)
            bcgnet = bcgnet_output_vhdr(config.paths.bcgnet_root, bids_id, src)
            triples.append(
                RecordingTriple(
                    bids_id=bids_id,
                    str_sub=str_sub,
                    idx_run=idx,
                    stem=src.stem,
                    fastr_vhdr=src,
                    aas_vhdr=aas if aas.is_file() else None,
                    bcgnet_vhdr=bcgnet if bcgnet.is_file() else None,
                )
            )
    return triples
