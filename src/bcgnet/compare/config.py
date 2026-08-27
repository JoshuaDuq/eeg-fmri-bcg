"""YAML for running and/or comparing AAS and BCGNet."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

from bcg_correction.bcg_config import DetectorConfig

from ..aas_batch import AasSettings
from ..config import ConfigurationError


@dataclass(frozen=True, slots=True)
class ComparePaths:
    fastr_root: Path
    aas_root: Path
    bcgnet_root: Path
    output_root: Path


@dataclass(frozen=True, slots=True)
class RunFlags:
    aas: bool
    bcgnet: bool


@dataclass(frozen=True, slots=True)
class PlotSettings:
    channel: str
    epoch_start_seconds: float
    epoch_seconds: float
    psd_max_hz: float


@dataclass(frozen=True, slots=True)
class CompareConfig:
    paths: ComparePaths
    run: RunFlags
    aas: AasSettings
    bcgnet_config: Path | None
    plot: PlotSettings
    include: tuple[str, ...]
    exclude: tuple[str, ...]


_TOP = frozenset({"paths", "run", "aas", "bcgnet_config", "plot", "subjects"})
_PATH_KEYS = frozenset({"fastr_root", "aas_root", "bcgnet_root", "output_root"})
_RUN_KEYS = frozenset({"aas", "bcgnet"})
_PLOT_KEYS = frozenset(
    {"channel", "epoch_start_seconds", "epoch_seconds", "psd_max_hz"}
)
_SUBJECT_KEYS = frozenset({"include", "exclude"})
_AAS_KEYS = frozenset(
    {
        "method",
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
    unknown = sorted(str(key) for key in document if key not in _TOP)
    if unknown:
        raise ConfigurationError(
            f"unknown field(s) in configuration: {', '.join(unknown)}"
        )
    for key in ("paths", "run", "aas", "plot", "subjects"):
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
    aas = _mapping(document["aas"], "aas")
    _require_keys(aas, _AAS_KEYS, "aas")
    detector = _mapping(aas["detector"], "aas.detector")
    _require_keys(detector, _DETECTOR_KEYS, "aas.detector")

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
    window = _two_floats(aas, "window_seconds")
    return CompareConfig(
        paths=ComparePaths(
            fastr_root=_path(paths, "fastr_root", base),
            aas_root=_path(paths, "aas_root", base),
            bcgnet_root=_path(paths, "bcgnet_root", base),
            output_root=_path(paths, "output_root", base),
        ),
        run=RunFlags(aas=_bool(run, "aas"), bcgnet=run_bcgnet),
        aas=AasSettings(
            method=_string(aas, "method"),
            window_seconds=window,
            ecg_to_bcg_delay_seconds=float(aas["ecg_to_bcg_delay_seconds"]),
            aas_neighbor_count=int(aas["aas_neighbor_count"]),
            pca_obs_components=int(aas["pca_obs_components"]),
            maximum_residual_ratio=float(aas["maximum_residual_ratio"]),
            overwrite=_bool(aas, "overwrite"),
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
    )


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{field} must be a mapping")
    return value


def _require_keys(
    values: Mapping[str, object], expected: frozenset[str], field: str
) -> None:
    unknown = sorted(str(key) for key in values if key not in expected)
    if unknown:
        raise ConfigurationError(
            f"unknown field(s) in {field}: {', '.join(unknown)}"
        )
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
