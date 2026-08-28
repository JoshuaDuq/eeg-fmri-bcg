"""A baseline recording is discovered and staged, but never counted as a run."""

from pathlib import Path

import yaml

from bcgnet.cohort import discover_subjects, run_count, stage_subject
from bcgnet.config import load_config

_HEADER = (
    "Brain Vision Data Exchange Header File Version 1.0\n"
    "DataFile=original.eeg\n"
    "MarkerFile=original.vmrk\n" + ("x" * 120)
)


def _cohort(tmp_path: Path, *names: str, naming: dict | None = None) -> Path:
    folder = tmp_path / "fastr" / "sub-0000"
    folder.mkdir(parents=True)
    for name in names:
        (folder / name).write_text(_HEADER, encoding="utf-8")
    document = {
        "paths": {
            "fastr_root": str(tmp_path / "fastr"),
            "output_root": str(tmp_path / "out"),
        },
        "compute": {"workers": 1, "cpu_count": 2, "threads_per_worker": "auto"},
        "training": {
            "num_epochs": 80,
            "es_patience": 12,
            "batch_size": 16,
            "learning_rate": 0.001,
            "random_seed": 1997,
            "architecture": "default_rnn_model",
            "overwrite": True,
            "resume": True,
            "save_model": True,
            "save_data": True,
            "save_figures": True,
        },
        "preprocess": {
            "new_fs": 100,
            "len_epoch": 3,
            "mad_threshold": 5,
            "per_training": 0.7,
            "per_valid": 0.15,
            "per_test": 0.15,
            "ecg_channel": "ECG",
        },
        "subjects": {"include": [], "exclude": []},
    }
    if naming is not None:
        document["naming"] = naming
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path


#: sub-0000 as it exists on disk: a baseline, and no run 1.
_SUB_0000 = (
    "BaselineEEG_sub0000_2026-02-09_10h56.55.966_fastr.vhdr",
    *(
        f"ThermalPainEEGFMRI_run{n}_sub0000_2026-02-09_11h.vhdr"
        for n in (2, 3, 4, 5, 6)
    ),
)


def test_discovery_labels_each_recording_from_its_filename(tmp_path):
    config = load_config(_cohort(tmp_path, *_SUB_0000))
    (spec,) = discover_subjects(config)
    assert [item["label"] for item in spec["recordings"]] == [
        "BaselineEEG",
        "run2",
        "run3",
        "run4",
        "run5",
        "run6",
    ]
    assert [item["run"] for item in spec["recordings"]] == [None, 2, 3, 4, 5, 6]


def test_baseline_is_not_counted_as_a_run(tmp_path):
    config = load_config(_cohort(tmp_path, *_SUB_0000))
    (spec,) = discover_subjects(config)
    assert len(spec["recordings"]) == 6
    assert run_count(spec) == 5


def test_a_missing_run_one_is_never_invented(tmp_path):
    config = load_config(_cohort(tmp_path, *_SUB_0000))
    (spec,) = discover_subjects(config)
    assert all(item["run"] != 1 for item in spec["recordings"])
    assert "run1" not in {item["label"] for item in spec["recordings"]}


def test_staged_filenames_carry_the_label_not_a_position(tmp_path):
    config = load_config(_cohort(tmp_path, *_SUB_0000))
    (spec,) = discover_subjects(config)
    output_root = tmp_path / "out"
    stage_subject(spec, output_root)
    staged = sorted(
        path.name
        for path in (output_root / "staged" / "raw_data" / "sub0000").glob("*.vhdr")
    )
    assert staged == [
        "sub0000_BaselineEEG_raw.vhdr",
        "sub0000_run2_raw.vhdr",
        "sub0000_run3_raw.vhdr",
        "sub0000_run4_raw.vhdr",
        "sub0000_run5_raw.vhdr",
        "sub0000_run6_raw.vhdr",
    ]


def test_staged_header_points_at_its_own_sidecars(tmp_path):
    config = load_config(_cohort(tmp_path, *_SUB_0000))
    (spec,) = discover_subjects(config)
    stage_subject(spec, tmp_path / "out")
    staged = tmp_path / "out" / "staged" / "raw_data" / "sub0000"
    text = (staged / "sub0000_run2_raw.vhdr").read_text(encoding="utf-8")
    assert "DataFile=sub0000_run2_raw.eeg" in text
    assert "MarkerFile=sub0000_run2_raw.vmrk" in text


def test_a_study_with_other_task_names_is_labelled_by_its_own_pattern(tmp_path):
    config = load_config(
        _cohort(
            tmp_path,
            "acquisition_rest_sub0000.vhdr",
            "acquisition_S03_sub0000.vhdr",
            naming={"run_pattern": r"_S(\d+)_"},
        )
    )
    (spec,) = discover_subjects(config)
    assert [item["label"] for item in spec["recordings"]] == [
        "acquisition",
        "run3",
    ]
    assert run_count(spec) == 1
