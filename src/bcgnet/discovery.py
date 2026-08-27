"""Shared FASTR subject / run discovery."""

from __future__ import annotations

import re
from pathlib import Path


def list_vhdrs(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(
        path
        for path in folder.glob("*.vhdr")
        if not path.name.startswith("._") and path.stat().st_size > 100
    )


def run_sort_key(path: Path):
    name = path.name
    if name.startswith("BaselineEEG"):
        return (0, 0, name)
    match = re.search(r"run(\d+)", name, re.I)
    run_n = int(match.group(1)) if match else 99
    return (1, run_n, name)


def iter_subjects(
    root: Path,
    *,
    include: tuple[str, ...] = (),
    exclude: tuple[str, ...] = (),
) -> list[tuple[str, str, list[Path]]]:
    """Return (bids_id, str_sub, fastr_vhdrs) for each subject folder."""
    subjects: list[tuple[str, str, list[Path]]] = []
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
        vhdrs = sorted(list_vhdrs(sub_dir), key=run_sort_key)
        if not vhdrs:
            continue
        subjects.append((bids_id, bids_id.replace("-", ""), vhdrs))
    return subjects
