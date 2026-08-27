"""Resolve which span of an input recording a correction run emits."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

_SUPPORTED_MODES = frozenset({"none", "first_to_last_volume"})


class WindowError(ValueError):
    """Raised when an output window cannot be resolved."""


@dataclass(frozen=True, slots=True)
class OutputWindow:
    """Zero-based, half-open bounds of the emitted span in input samples."""

    start: int
    stop: int

    def __post_init__(self) -> None:
        for name in ("start", "stop"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise WindowError(f"{name} must be an integer sample index")
        if self.start < 0 or self.stop <= self.start:
            raise WindowError("output window must be a non-empty forward span")

    @property
    def length(self) -> int:
        return self.stop - self.start


def resolve_output_window(
    volume_starts: npt.ArrayLike,
    *,
    mode: str,
    input_sample_count: int,
) -> OutputWindow:
    """Resolve the emitted span for one trim mode.

    ``first_to_last_volume`` reproduces the trim these recordings were prepared
    with externally: the first volume marker through the last one, inclusive.
    Correction still runs over the whole recording, so the volumes at either end
    of this span keep the neighbours and the complete epochs they need.
    """
    if isinstance(input_sample_count, bool) or not isinstance(
        input_sample_count, int
    ):
        raise WindowError("input sample count must be a positive integer")
    if input_sample_count < 1:
        raise WindowError("input sample count must be a positive integer")
    if mode not in _SUPPORTED_MODES:
        raise WindowError(f"unsupported trim mode: {mode!r}")
    if mode == "none":
        return OutputWindow(start=0, stop=input_sample_count)

    starts = np.asarray(volume_starts)
    if starts.ndim != 1 or starts.size < 1:
        raise WindowError("volume starts must be a non-empty one-dimensional array")
    if not np.issubdtype(starts.dtype, np.integer):
        raise WindowError("volume starts must contain integer sample positions")
    start = int(starts[0])
    stop = int(starts[-1]) + 1
    if start < 0 or stop > input_sample_count:
        raise WindowError("resolved output window falls outside the recording")
    return OutputWindow(start=start, stop=stop)
