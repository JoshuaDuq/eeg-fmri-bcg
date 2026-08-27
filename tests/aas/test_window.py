import numpy as np
import pytest

from bcg_correction.window import OutputWindow, WindowError, resolve_output_window


def test_none_mode_covers_the_whole_recording() -> None:
    window = resolve_output_window(
        np.array([0, 4500], dtype=np.int64),
        mode="none",
        input_sample_count=10_000,
    )

    assert (window.start, window.stop) == (0, 10_000)
    assert window.length == 10_000


def test_first_to_last_volume_spans_marker_to_marker_inclusive() -> None:
    window = resolve_output_window(
        np.array([142_276, 3_107_776], dtype=np.int64),
        mode="first_to_last_volume",
        input_sample_count=3_160_200,
    )

    assert (window.start, window.stop) == (142_276, 3_107_777)
    assert window.length == 2_965_501


def test_already_trimmed_input_resolves_to_the_whole_recording() -> None:
    window = resolve_output_window(
        np.array([0, 4500, 9000], dtype=np.int64),
        mode="first_to_last_volume",
        input_sample_count=9001,
    )

    assert (window.start, window.stop) == (0, 9001)


def test_window_beyond_the_recording_is_rejected() -> None:
    with pytest.raises(WindowError, match="outside the recording"):
        resolve_output_window(
            np.array([0, 9000], dtype=np.int64),
            mode="first_to_last_volume",
            input_sample_count=5000,
        )


def test_unsupported_mode_is_rejected() -> None:
    with pytest.raises(WindowError, match="unsupported trim mode"):
        resolve_output_window(
            np.array([0], dtype=np.int64),
            mode="everything",
            input_sample_count=10,
        )


def test_empty_volume_starts_are_rejected() -> None:
    with pytest.raises(WindowError, match="non-empty"):
        resolve_output_window(
            np.empty(0, dtype=np.int64),
            mode="first_to_last_volume",
            input_sample_count=10,
        )


def test_window_bounds_must_form_a_forward_span() -> None:
    with pytest.raises(WindowError, match="forward span"):
        OutputWindow(start=10, stop=10)
