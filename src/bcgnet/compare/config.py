"""YAML for running and/or comparing the bounded arms and BCGNet."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from pathlib import Path

import yaml

from bcg_correction.bcg_config import DetectorConfig
from bcg_correction.evaluation import parse_evaluation
from bcgstudy.correction_batch import CorrectionSettings

from ..config import ConfigurationError
from ..config import run_pattern as _run_pattern
from .arms import CLEAN_ARMS, Arm


@dataclass(frozen=True, slots=True)
class ComparePaths:
    fastr_root: Path
    output_root: Path
    #: Where the method-vs-method pages go. Relative paths resolve against the
    #: folder holding the compare config, so the study's own ``experiments/``
    #: keeps the figures beside the configs that produced them.
    experiments_root: Path
    arm_roots: tuple[tuple[str, Path], ...]

    def root_for(self, arm: Arm) -> Path:
        """Return the folder holding ``arm``'s corrected recordings."""
        roots = dict(self.arm_roots)
        try:
            return roots[arm.key]
        except KeyError:
            raise ValueError(f"no configured root for arm {arm.key!r}") from None


@dataclass(frozen=True, slots=True)
class RunFlags:
    enabled_keys: frozenset[str]

    def enabled(self, arm: Arm) -> bool:
        """Whether ``bcg compare`` should generate ``arm`` before plotting."""
        if arm not in CLEAN_ARMS:
            raise ValueError(f"no run flag for arm {arm.key!r}") from None
        return arm.key in self.enabled_keys


@dataclass(frozen=True, slots=True)
class CompareCompute:
    """How much of the machine the bounded arms may use.

    Recordings are corrected independently, so this is simply how many of them
    run at once. One keeps the batch in a single process, as it always was.
    """

    workers: int


@dataclass(frozen=True, slots=True)
class PlotSettings:
    channel: str
    epoch_start_seconds: float
    epoch_seconds: float
    psd_max_hz: float


@dataclass(frozen=True, slots=True)
class CompareConfig:
    paths: ComparePaths
    compute: CompareCompute
    run: RunFlags
    correction: CorrectionSettings
    bcgnet_config: Path | None
    plot: PlotSettings
    include: tuple[str, ...]
    exclude: tuple[str, ...]
    #: Regex whose first group is the run number in a recording's filename.
    run_pattern: str


_TOP = frozenset(
    {
        "paths",
        "compute",
        "run",
        "correction",
        "bcgnet_config",
        "plot",
        "subjects",
        "naming",
    }
)
# Keys that moved, so a pre-PCA-OBS config fails with the fix rather than a
# bare "unknown field".
_RENAMED = {"aas": "correction"}
_ARM_PATH_KEYS = frozenset(f"{arm.key}_root" for arm in CLEAN_ARMS)
_PATH_KEYS = _ARM_PATH_KEYS | {"fastr_root", "output_root", "experiments_root"}
_RUN_KEYS = frozenset(arm.key for arm in CLEAN_ARMS)
_COMPUTE_KEYS = frozenset({"workers"})
_PLOT_KEYS = frozenset(
    {"channel", "epoch_start_seconds", "epoch_seconds", "psd_max_hz"}
)
_SUBJECT_KEYS = frozenset({"include", "exclude"})
_NAMING_KEYS = frozenset({"run_pattern"})
_CORRECTION_KEYS = frozenset(
    {
        "window_seconds",
        "ecg_to_bcg_delay_seconds",
        "aas_neighbor_count",
        "pca_obs_components",
        "evaluation",
        "maximum_gap_fraction",
        "overwrite",
        "detector",
    }
)
_DETECTOR_KEYS = frozenset(
    {
        "ecg_channel",
        "preprocessing_band_hz",
        "teager_emphasis_hz",
        "teager_smoothing_seconds",
        "template_window_seconds",
        "minimum_rr_seconds",
        "maximum_rr_seconds",
        "candidate_refractory_seconds",
        "candidate_prominence_mad",
        "correlation_threshold",
        "refinement_iterations",
    }
)


def _workers(compute: Mapping[str, object]) -> int:
    """Validate ``compute.workers``."""
    value = compute["workers"]
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ConfigurationError(
            "compute.workers must be an integer greater than or equal to 1"
        )
    return value


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{field} must be a mapping")
    return value


def _reject_unknown_keys(
    values: Mapping[str, object], expected: frozenset[str], field: str
) -> None:
    unknown = sorted(str(key) for key in values if key not in expected)
    if unknown:
        raise ConfigurationError(f"unknown field(s) in {field}: {', '.join(unknown)}")


def _require_keys(
    values: Mapping[str, object], expected: frozenset[str], field: str
) -> None:
    _reject_unknown_keys(values, expected, field)
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


def _path(values: Mapping[str, object], name: str, base: Path) -> Path:
    path = Path(_string(values, name)).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _ordered_interval(
    values: Mapping[str, object], name: str, *, positive: bool = False
) -> tuple[float, float]:
    value = values[name]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ConfigurationError(f"{name} must be two numbers")
    if len(value) != 2:
        raise ConfigurationError(f"{name} must contain exactly two numbers")
    interval = (
        _finite_number_value(value[0], name),
        _finite_number_value(value[1], name),
    )
    if interval[0] >= interval[1]:
        raise ConfigurationError(f"{name} must be increasing")
    if positive and interval[0] <= 0.0:
        raise ConfigurationError(f"{name} must contain positive values")
    return interval


def _string_list(values: Mapping[str, object], name: str) -> tuple[str, ...]:
    value = values[name]
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ConfigurationError(f"{name} must be a list of strings")
    if any(not isinstance(item, str) or not item for item in value):
        raise ConfigurationError(f"{name} must be a list of nonempty strings")
    return tuple(value)


def _finite_number_value(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ConfigurationError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ConfigurationError(f"{name} must be a finite number")
    return number


def _positive_number(values: Mapping[str, object], name: str) -> float:
    number = _finite_number_value(values[name], name)
    if number <= 0.0:
        raise ConfigurationError(f"{name} must be a finite positive number")
    return number


def _nonnegative_number(values: Mapping[str, object], name: str) -> float:
    number = _finite_number_value(values[name], name)
    if number < 0.0:
        raise ConfigurationError(f"{name} must be a finite nonnegative number")
    return number


def _fraction(values: Mapping[str, object], name: str) -> float:
    number = _finite_number_value(values[name], name)
    if number <= 0.0 or number > 1.0:
        raise ConfigurationError(f"{name} must be a number in (0, 1]")
    return number


def _integer(values: Mapping[str, object], name: str, *, minimum: int) -> int:
    value = values[name]
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigurationError(
            f"{name} must be an integer greater than or equal to {minimum}"
        )
    return value


def _even_integer(values: Mapping[str, object], name: str, *, minimum: int) -> int:
    value = _integer(values, name, minimum=minimum)
    if value % 2:
        raise ConfigurationError(f"{name} must be even")
    return value


def load_compare_config(path: str | Path) -> CompareConfig:
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
    if not isinstance(document, Mapping):
        raise ConfigurationError("configuration must be a mapping")
    for old, new in _RENAMED.items():
        if old in document:
            raise ConfigurationError(
                f"field {old!r} was renamed to {new!r}; the correction settings "
                "are now shared by every bounded arm, which are selected "
                "by run flags instead of a method string"
            )
    unknown = sorted(str(key) for key in document if key not in _TOP)
    if unknown:
        raise ConfigurationError(
            f"unknown field(s) in configuration: {', '.join(unknown)}"
        )
    for key in ("paths", "compute", "run", "correction", "plot", "subjects"):
        if key not in document:
            raise ConfigurationError(f"missing required field: {key}")

    base = config_path.parent
    paths = _mapping(document["paths"], "paths")
    _require_keys(paths, _PATH_KEYS, "paths")
    run = _mapping(document["run"], "run")
    _require_keys(run, _RUN_KEYS, "run")
    plot = _mapping(document["plot"], "plot")
    _require_keys(plot, _PLOT_KEYS, "plot")
    subjects = _mapping(document["subjects"], "subjects")
    _require_keys(subjects, _SUBJECT_KEYS, "subjects")
    # ``naming`` is optional, so adding it never invalidates an existing config.
    naming = _mapping(document.get("naming", {}), "naming")
    _reject_unknown_keys(naming, _NAMING_KEYS, "naming")
    compute = _mapping(document["compute"], "compute")
    _require_keys(compute, _COMPUTE_KEYS, "compute")
    correction = _mapping(document["correction"], "correction")
    _require_keys(correction, _CORRECTION_KEYS, "correction")
    detector = _mapping(correction["detector"], "correction.detector")
    _require_keys(detector, _DETECTOR_KEYS, "correction.detector")

    run_flags = frozenset(key for key in _RUN_KEYS if _bool(run, key))
    run_bcgnet = "bcgnet" in run_flags
    bcgnet_config = document.get("bcgnet_config")
    if run_bcgnet:
        if not isinstance(bcgnet_config, str) or not bcgnet_config:
            raise ConfigurationError(
                "bcgnet_config is required when run.bcgnet is true"
            )
        bcgnet_path = Path(bcgnet_config).expanduser()
        if not bcgnet_path.is_absolute():
            bcgnet_path = base / bcgnet_path
        bcgnet_path = bcgnet_path.resolve()
    else:
        bcgnet_path = None

    band = _ordered_interval(detector, "preprocessing_band_hz", positive=True)
    template = _ordered_interval(detector, "template_window_seconds")
    window = _ordered_interval(correction, "window_seconds")
    minimum_rr = _positive_number(detector, "minimum_rr_seconds")
    maximum_rr = _positive_number(detector, "maximum_rr_seconds")
    if minimum_rr >= maximum_rr:
        raise ConfigurationError(
            "minimum_rr_seconds must be less than maximum_rr_seconds"
        )
    return CompareConfig(
        compute=CompareCompute(workers=_workers(compute)),
        paths=ComparePaths(
            fastr_root=_path(paths, "fastr_root", base),
            output_root=_path(paths, "output_root", base),
            experiments_root=_path(paths, "experiments_root", base),
            arm_roots=tuple(
                (arm.key, _path(paths, f"{arm.key}_root", base)) for arm in CLEAN_ARMS
            ),
        ),
        run=RunFlags(enabled_keys=run_flags),
        correction=CorrectionSettings(
            window_seconds=window,
            ecg_to_bcg_delay_seconds=_nonnegative_number(
                correction, "ecg_to_bcg_delay_seconds"
            ),
            aas_neighbor_count=_even_integer(
                correction, "aas_neighbor_count", minimum=2
            ),
            pca_obs_components=_integer(correction, "pca_obs_components", minimum=1),
            evaluation=parse_evaluation(correction["evaluation"]),
            maximum_gap_fraction=_fraction(correction, "maximum_gap_fraction"),
            overwrite=_bool(correction, "overwrite"),
            detector=DetectorConfig(
                ecg_channel=_string(detector, "ecg_channel"),
                preprocessing_band_hz=band,
                teager_emphasis_hz=_positive_number(detector, "teager_emphasis_hz"),
                teager_smoothing_seconds=_positive_number(
                    detector, "teager_smoothing_seconds"
                ),
                template_window_seconds=template,
                minimum_rr_seconds=minimum_rr,
                maximum_rr_seconds=maximum_rr,
                candidate_refractory_seconds=_positive_number(
                    detector, "candidate_refractory_seconds"
                ),
                candidate_prominence_mad=_positive_number(
                    detector, "candidate_prominence_mad"
                ),
                correlation_threshold=_fraction(detector, "correlation_threshold"),
                refinement_iterations=_integer(
                    detector, "refinement_iterations", minimum=1
                ),
            ),
        ),
        bcgnet_config=bcgnet_path,
        plot=PlotSettings(
            channel=_string(plot, "channel"),
            epoch_start_seconds=_nonnegative_number(plot, "epoch_start_seconds"),
            epoch_seconds=_positive_number(plot, "epoch_seconds"),
            psd_max_hz=_positive_number(plot, "psd_max_hz"),
        ),
        include=_string_list(subjects, "include"),
        exclude=_string_list(subjects, "exclude"),
        run_pattern=_run_pattern(naming),
    )
