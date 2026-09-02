import csv
import json

import pytest

import bcgnet.cohort as cohort
from bcgnet.cohort import write_cohort_summary


def test_initial_result_describes_every_recording() -> None:
    spec = {
        "bids_id": "sub-0001",
        "str_sub": "sub0001",
        "recordings": [
            {"label": "BaselineEEG", "run": None, "stem": "baseline"},
            {"label": "run2", "run": 2, "stem": "task"},
        ],
    }

    result = cohort._initial_result(spec)

    assert result["status"] == "error"
    assert result["n_runs"] == 1
    assert result["n_recordings"] == 2
    assert result["recordings"][1] == {
        "label": "run2",
        "run": 2,
        "stem": "task",
    }


def test_resume_rejects_a_corrupt_subject_result(tmp_path) -> None:
    result_path = tmp_path / "sub-0001.json"
    result_path.write_text("not json", encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        cohort._completed_result(result_path, resume=True)


@pytest.mark.parametrize("reverse", [False, True])
def test_cohort_summary_schema_does_not_depend_on_completion_order(
    tmp_path, reverse: bool
) -> None:
    failed = {
        "bids_id": "sub-failed",
        "status": "error",
        "n_runs": 1,
        "runtime_seconds": 2.0,
    }
    succeeded = {
        "bids_id": "sub-ok",
        "status": "ok",
        "end_epoch": 3,
        "runtime_seconds": 4.0,
        "metrics": [
            {
                "label": "run1",
                "run": 1,
                "stem": "recording,with-comma",
                "n_good": 5,
                "rms_raw": 1.0,
                "rms_bcgnet": 0.5,
                "bands": [],
            }
        ],
    }
    results = [failed, succeeded]
    if reverse:
        results.reverse()

    output = write_cohort_summary(results, tmp_path)

    with output.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_subject = {row["bids_id"]: row for row in rows}
    assert by_subject["sub-failed"]["n_runs"] == "1"
    assert by_subject["sub-ok"]["stem"] == "recording,with-comma"
    assert "rms_bcgnet" in by_subject["sub-failed"]


def test_empty_cohort_summary_replaces_stale_csv(tmp_path) -> None:
    summary = tmp_path / "cohort_summary.csv"
    summary.write_text("stale,data\n1,2\n", encoding="utf-8")

    output = write_cohort_summary([], tmp_path)

    with output.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == list(cohort._COHORT_SUMMARY_FIELDS)
        assert list(reader) == []
