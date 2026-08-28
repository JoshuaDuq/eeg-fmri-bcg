"""Strict YAML configuration for the BCGNet cohort pipeline."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from pathlib import Path

import yaml


class ConfigurationError(ValueError):
    """Raised when a YAML configuration does not describe a valid run."""


@dataclass(frozen=True, slots=True)
class PathConfig:
    fastr_root: Path
    output_root: Path


@dataclass(frozen=True, slots=True)
class ComputeConfig:
    workers: int
    cpu_count: int
    threads_per_worker: int


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    num_epochs: int
    es_patience: int
    batch_size: int
    learning_rate: float
    random_seed: int
    architecture: str
    overwrite: bool
    resume: bool
    save_model: bool
    save_data: bool
    save_figures: bool


@dataclass(frozen=True, slots=True)
class PreprocessConfig:
    new_fs: int
    len_epoch: float
    mad_threshold: float
    per_training: float
    per_valid: float
    per_test: float
    ecg_channel: str


@dataclass(frozen=True, slots=True)
class SubjectConfig:
    include: tuple[str, ...]
    exclude: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BCGNetConfig:
    paths: PathConfig
    compute: ComputeConfig
    training: TrainingConfig
    preprocess: PreprocessConfig
    subjects: SubjectConfig


_TOP = frozenset({"paths", "compute", "training", "preprocess", "subjects"})
_PATH_KEYS = frozenset({"fastr_root", "output_root"})
_COMPUTE_KEYS = frozenset({"workers", "cpu_count", "threads_per_worker"})
_TRAINING_KEYS = frozenset(
    {
        "num_epochs",
        "es_patience",
        "batch_size",
        "learning_rate",
        "random_seed",
        "architecture",
        "overwrite",
        "resume",
        "save_model",
        "save_data",
        "save_figures",
    }
)
_PREPROCESS_KEYS = frozenset(
    {
        "new_fs",
        "len_epoch",
        "mad_threshold",
        "per_training",
        "per_valid",
        "per_test",
        "ecg_channel",
    }
)
_SUBJECT_KEYS = frozenset({"include", "exclude"})


def load_config(path: str | Path) -> BCGNetConfig:
    config_path = Path(path).expanduser().resolve()
    try:
        document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ConfigurationError(
            f"configuration file does not exist: {config_path}"
        ) from error
    except yaml.YAMLError as error:
        raise ConfigurationError(
            f"invalid YAML in configuration: {config_path}"
        ) from error

    root = _mapping(document, "configuration")
    _reject_unknown(root, _TOP, "configuration")
    _require(root, _TOP, "configuration")
    base = config_path.parent

    paths = _section(root, "paths", _PATH_KEYS)
    compute = _section(root, "compute", _COMPUTE_KEYS)
    training = _section(root, "training", _TRAINING_KEYS)
    preprocess = _section(root, "preprocess", _PREPROCESS_KEYS)
    subjects = _section(root, "subjects", _SUBJECT_KEYS)

    workers = _integer(compute, "workers", minimum=1)
    cpu_count = _integer(compute, "cpu_count", minimum=1)
    threads = compute.get("threads_per_worker")
    if threads is None or threads == "auto":
        threads_per_worker = max(1, cpu_count // workers)
    else:
        threads_per_worker = _integer(compute, "threads_per_worker", minimum=1)

    per_training = _fraction(preprocess, "per_training")
    per_valid = _fraction(preprocess, "per_valid")
    per_test = _fraction(preprocess, "per_test")
    split_sum = per_training + per_valid + per_test
    if abs(split_sum - 1.0) > 1e-9:
        raise ConfigurationError(
            "preprocess.per_training, per_valid, and per_test must sum to 1"
        )
    if workers > cpu_count:
        raise ConfigurationError("compute.workers cannot exceed compute.cpu_count")

    return BCGNetConfig(
        paths=PathConfig(
            fastr_root=_path_value(paths, "fastr_root", base),
            output_root=_path_value(paths, "output_root", base),
        ),
        compute=ComputeConfig(
            workers=workers,
            cpu_count=cpu_count,
            threads_per_worker=threads_per_worker,
        ),
        training=TrainingConfig(
            num_epochs=_integer(training, "num_epochs", minimum=1),
            es_patience=_integer(training, "es_patience", minimum=1),
            batch_size=_integer(training, "batch_size", minimum=1),
            learning_rate=_positive_number(training, "learning_rate"),
            random_seed=_integer(training, "random_seed", minimum=0),
            architecture=_string(training, "architecture"),
            overwrite=_bool(training, "overwrite"),
            resume=_bool(training, "resume"),
            save_model=_bool(training, "save_model"),
            save_data=_bool(training, "save_data"),
            save_figures=_bool(training, "save_figures"),
        ),
        preprocess=PreprocessConfig(
            new_fs=_integer(preprocess, "new_fs", minimum=1),
            len_epoch=_positive_number(preprocess, "len_epoch"),
            mad_threshold=_positive_number(preprocess, "mad_threshold"),
            per_training=per_training,
            per_valid=per_valid,
            per_test=per_test,
            ecg_channel=_string(preprocess, "ecg_channel"),
        ),
        subjects=SubjectConfig(
            include=_string_list(subjects, "include"),
            exclude=_string_list(subjects, "exclude"),
        ),
    )


def _section(
    root: Mapping[str, object],
    name: str,
    expected: frozenset[str],
) -> Mapping[str, object]:
    values = _mapping(root[name], name)
    _reject_unknown(values, expected, name)
    required = expected
    if name == "compute":
        required = expected - {"threads_per_worker"}
        if "threads_per_worker" not in values:
            values = dict(values)
            values["threads_per_worker"] = "auto"
    _require(values, required, name)
    return values


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{field} must be a mapping")
    return value


def _reject_unknown(
    values: Mapping[str, object],
    expected: frozenset[str],
    field: str,
) -> None:
    unknown = sorted(str(key) for key in values if key not in expected)
    if unknown:
        raise ConfigurationError(
            f"unknown field(s) in {field}: {', '.join(unknown)}"
        )


def _require(
    values: Mapping[str, object],
    expected: frozenset[str],
    field: str,
) -> None:
    for key in sorted(expected):
        if key not in values:
            raise ConfigurationError(f"missing required field: {field}.{key}")


def _string(values: Mapping[str, object], name: str) -> str:
    value = values[name]
    if not isinstance(value, str) or not value:
        raise ConfigurationError(f"{name} must be a nonempty string")
    return value


def _bool(values: Mapping[str, object], name: str) -> bool:
    value = values[name]
    if not isinstance(value, bool):
        raise ConfigurationError(f"{name} must be a boolean")
    return value


def _integer(values: Mapping[str, object], name: str, *, minimum: int) -> int:
    value = values[name]
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigurationError(
            f"{name} must be an integer greater than or equal to {minimum}"
        )
    return value


def _positive_number(values: Mapping[str, object], name: str) -> float:
    value = values[name]
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ConfigurationError(f"{name} must be a finite positive number")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ConfigurationError(f"{name} must be a finite positive number")
    return number


def _fraction(values: Mapping[str, object], name: str) -> float:
    value = values[name]
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ConfigurationError(f"{name} must be a number in (0, 1]")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0 or number > 1.0:
        raise ConfigurationError(f"{name} must be a number in (0, 1]")
    return number


def _path_value(
    values: Mapping[str, object],
    name: str,
    base: Path,
) -> Path:
    value = _string(values, name)
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _string_list(values: Mapping[str, object], name: str) -> tuple[str, ...]:
    value = values[name]
    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ConfigurationError(f"{name} must be a list of strings")
    if any(not isinstance(item, str) or not item for item in value):
        raise ConfigurationError(f"{name} must be a list of nonempty strings")
    return tuple(value)
