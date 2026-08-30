"""YAML for running and/or comparing the bounded arms and BCGNet."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

from bcg_correction.bcg_config import DetectorConfig

from ..config import ConfigurationError
from ..config import run_pattern as _run_pattern
from ..correction_batch import CorrectionSettings
from .arms import AAS, BCGNET, PCA_OBS, Arm


@dataclass(frozen=True, slots=True)
class ComparePaths:
    fastr_root: Path
    aas_root: Path
    pca_obs_root: Path
    bcgnet_root: Path
    output_root: Path

    def root_for(self, arm: Arm) -> Path:
        """Return the folder holding ``arm``'s corrected recordings."""
        roots = {
            AAS.key: self.aas_root,
            PCA_OBS.key: self.pca_obs_root,
            BCGNET.key: self.bcgnet_root,
        }
        try:
            return roots[arm.key]
        except KeyError:
            raise ValueError(f"no configured root for arm {arm.key!r}") from None


@dataclass(frozen=True, slots=True)
class RunFlags:
    aas: bool
    pca_obs: bool
    bcgnet: bool

    def enabled(self, arm: Arm) -> bool:
        """Whether ``bcgnet compare`` should generate ``arm`` before plotting."""
        flags = {
            AAS.key: self.aas,
            PCA_OBS.key: self.pca_obs,
            BCGNET.key: self.bcgnet,
        }
        try:
            return flags[arm.key]
        except KeyError:
            raise ValueError(f"no run flag for arm {arm.key!r}") from None


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
_PATH_KEYS = frozenset(
    {"fastr_root", "aas_root", "pca_obs_root", "bcgnet_root", "output_root"}
)
_RUN_KEYS = frozenset({"aas", "pca_obs", "bcgnet"})
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
        "maximum_residual_ratio",
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
                "are now shared by the aas and pca_obs arms, which are selected "
                "by run flags instead of a method string"
            )
    unknown = sorted(str(key) for key in document if key not in _TOP)
    if unknown:
        raise ConfigurationError(
            f"unknown field(s) in configuration: {', '.join(unknown)}"
        )
    for key in ("paths", "run", "correction", "plot", "subjects"):
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
    # ``compute`` likewise: absent means the serial batch this file used to run.
    compute = _mapping(document.get("compute", {}), "compute")
    _reject_unknown_keys(compute, _COMPUTE_KEYS, "compute")
    correction = _mapping(document["correction"], "correction")
    _require_keys(correction, _CORRECTION_KEYS, "correction")
    detector = _mapping(correction["detector"], "correction.detector")
    _require_keys(detector, _DETECTOR_KEYS, "correction.detector")

    run_bcgnet = _bool(run, "bcgnet")
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

    band = _two_floats(detector, "preprocessing_band_hz")
    template = _two_floats(detector, "template_window_seconds")
    window = _two_floats(correction, "window_seconds")
    return CompareConfig(
        compute=CompareCompute(workers=_workers(compute)),
        paths=ComparePaths(
            fastr_root=_path(paths, "fastr_root", base),
            aas_root=_path(paths, "aas_root", base),
            pca_obs_root=_path(paths, "pca_obs_root", base),
            bcgnet_root=_path(paths, "bcgnet_root", base),
            output_root=_path(paths, "output_root", base),
        ),
        run=RunFlags(
            aas=_bool(run, "aas"),
            pca_obs=_bool(run, "pca_obs"),
            bcgnet=run_bcgnet,
        ),
        correction=CorrectionSettings(
            window_seconds=window,
            ecg_to_bcg_delay_seconds=float(
                correction["ecg_to_bcg_delay_seconds"]
            ),
            aas_neighbor_count=int(correction["aas_neighbor_count"]),
            pca_obs_components=int(correction["pca_obs_components"]),
            maximum_residual_ratio=float(correction["maximum_residual_ratio"]),
            overwrite=_bool(correction, "overwrite"),
            detector=DetectorConfig(
                ecg_channel=_string(detector, "ecg_channel"),
                preprocessing_band_hz=band,
                teager_emphasis_hz=float(detector["teager_emphasis_hz"]),
                teager_smoothing_seconds=float(
                    detector["teager_smoothing_seconds"]
                ),
                template_window_seconds=template,
                minimum_rr_seconds=float(detector["minimum_rr_seconds"]),
                maximum_rr_seconds=float(detector["maximum_rr_seconds"]),
                candidate_refractory_seconds=float(
                    detector["candidate_refractory_seconds"]
                ),
                candidate_prominence_mad=float(
                    detector["candidate_prominence_mad"]
                ),
                correlation_threshold=float(detector["correlation_threshold"]),
                refinement_iterations=int(detector["refinement_iterations"]),
            ),
        ),
        bcgnet_config=bcgnet_path,
        plot=PlotSettings(
            channel=_string(plot, "channel"),
            epoch_start_seconds=float(plot["epoch_start_seconds"]),
            epoch_seconds=float(plot["epoch_seconds"]),
            psd_max_hz=float(plot["psd_max_hz"]),
        ),
        include=_string_list(subjects, "include"),
        exclude=_string_list(subjects, "exclude"),
        run_pattern=_run_pattern(naming),
    )


def _workers(compute: Mapping[str, object]) -> int:
    """Validate ``compute.workers``, defaulting to the historical serial batch."""
    if "workers" not in compute:
        return 1
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
        raise ConfigurationError(
            f"unknown field(s) in {field}: {', '.join(unknown)}"
        )


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


def _two_floats(
    values: Mapping[str, object], name: str
) -> tuple[float, float]:
    value = values[name]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ConfigurationError(f"{name} must be two numbers")
    if len(value) != 2:
        raise ConfigurationError(f"{name} must contain exactly two numbers")
    return float(value[0]), float(value[1])


def _string_list(values: Mapping[str, object], name: str) -> tuple[str, ...]:
    value = values[name]
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ConfigurationError(f"{name} must be a list of strings")
    return tuple(str(item) for item in value)
