"""Method-neutral FASTR subject and recording discovery.

Each recording is labelled from its own filename rather than from its position
in the folder listing. A name carrying a run token -- ``run1``, ``run-02``,
``RUN_3`` -- is that run; a name carrying none, such as a baseline or resting
acquisition, is not a run at all. It is still discovered and processed, it is
just never counted, staged, or plotted as one.

Nothing here knows the task names used by any particular study.
``DEFAULT_RUN_PATTERN`` recognises the usual spellings, and a dataset that
numbers its runs some other way supplies its own regex through
``naming.run_pattern`` in the configuration.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

#: Matches ``run1``, ``run_2``, ``run-02``, ``RUN.3``. The lookbehind keeps the
#: token from matching inside a longer word, so neither the ``r`` of a
#: ``_fastr_`` pipeline suffix nor a task called ``prerun2`` reads as a run.
DEFAULT_RUN_PATTERN = r"(?<![A-Za-z])run[ _.-]?(\d+)"

_UNSAFE = re.compile(r"[^A-Za-z0-9]+")
_FALLBACK_LABEL = "recording"


@dataclass(frozen=True, slots=True)
class Recording:
    path: Path
    label: str
    run: int | None

    @property
    def is_run(self) -> bool:
        return self.run is not None

    @property
    def stem(self) -> str:
        return self.path.stem


def _non_run_label(stem: str) -> str:
    for candidate in (stem.split("_")[0], stem):
        safe = _UNSAFE.sub("", candidate)
        if safe:
            return safe
    return _FALLBACK_LABEL


def _unique(label: str, taken: set[str]) -> str:
    candidate = label
    suffix = 2
    while candidate in taken:
        candidate = f"{label}_{suffix}"
        suffix += 1
    taken.add(candidate)
    return candidate


def _sort_key(recording: Recording) -> tuple[int, int, str]:
    if recording.run is None:
        return (0, 0, recording.label)
    return (1, recording.run, recording.label)


def list_vhdrs(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(
        path
        for path in folder.glob("*.vhdr")
        if not path.name.startswith("._") and path.stat().st_size > 100
    )


def label_recordings(
    paths: list[Path],
    *,
    run_pattern: str = DEFAULT_RUN_PATTERN,
) -> list[Recording]:
    pattern = re.compile(run_pattern, re.IGNORECASE)
    taken: set[str] = set()
    recordings = []
    for path in sorted(paths, key=lambda item: item.name):
        match = pattern.search(path.stem)
        run = int(match.group(1)) if match else None
        base = f"run{run}" if run is not None else _non_run_label(path.stem)
        recordings.append(Recording(path=path, label=_unique(base, taken), run=run))
    return sorted(recordings, key=_sort_key)


def iter_subjects(
    root: Path,
    *,
    include: tuple[str, ...] = (),
    exclude: tuple[str, ...] = (),
    run_pattern: str = DEFAULT_RUN_PATTERN,
) -> list[tuple[str, str, list[Recording]]]:
    subjects: list[tuple[str, str, list[Recording]]] = []
    if not root.is_dir():
        return subjects
    for sub_dir in sorted(root.iterdir()):
        if not sub_dir.is_dir() or not sub_dir.name.startswith("sub-"):
            continue
        if sub_dir.name.startswith("._"):
            continue
        bids_id = sub_dir.name
        if include and bids_id not in include:
            if bids_id.replace("-", "") not in include:
                continue
        if bids_id in exclude:
            continue
        recordings = label_recordings(
            list_vhdrs(sub_dir), run_pattern=run_pattern
        )
        if not recordings:
            continue
        subjects.append((bids_id, bids_id.replace("-", ""), recordings))
    return subjects
