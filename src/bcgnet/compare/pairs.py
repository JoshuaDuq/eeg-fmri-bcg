"""Match one FASTR recording to whichever corrected arms exist on disk."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from bcgstudy.correction_batch import correction_output_vhdr
from bcgstudy.discovery import iter_subjects

from ..export import bcgnet_output_vhdr
from .arms import BCGNET, CLEAN_ARMS, Arm
from .config import CompareConfig


@dataclass(frozen=True, slots=True)
class RecordingSet:
    bids_id: str
    str_sub: str
    label: str
    run: int | None
    stem: str
    fastr_vhdr: Path
    cleaned_vhdr: Mapping[str, Path]

    def has(self, arm: Arm) -> bool:
        return arm.key in self.cleaned_vhdr


def arm_output_vhdr(config: CompareConfig, bids_id: str, src: Path, arm: Arm) -> Path:
    root = config.paths.root_for(arm)
    if arm is BCGNET:
        return bcgnet_output_vhdr(root, bids_id, src)
    return correction_output_vhdr(root, bids_id, src, arm=arm)


def pair_recordings(config: CompareConfig) -> list[RecordingSet]:
    recordings: list[RecordingSet] = []
    for bids_id, str_sub, found in iter_subjects(
        config.paths.fastr_root,
        include=config.include,
        exclude=config.exclude,
        run_pattern=config.run_pattern,
    ):
        for recording in found:
            src = recording.path
            cleaned: dict[str, Path] = {}
            for arm in CLEAN_ARMS:
                candidate = arm_output_vhdr(config, bids_id, src, arm)
                if candidate.is_file():
                    cleaned[arm.key] = candidate
            recordings.append(
                RecordingSet(
                    bids_id=bids_id,
                    str_sub=str_sub,
                    label=recording.label,
                    run=recording.run,
                    stem=recording.stem,
                    fastr_vhdr=src,
                    cleaned_vhdr=cleaned,
                )
            )
    return recordings
