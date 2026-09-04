"""Strict configuration for independent cardiac detection and BCG benchmarks."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from pathlib import Path

import yaml

from .bcg import METHODS
from .config import ConfigurationError
from .evaluation import EvaluationSettings, parse_evaluation


@dataclass(frozen=True, slots=True)
class DetectorConfig:
    """Parameters for deterministic ECG R-peak detection."""

    ecg_channel: str
    preprocessing_band_hz: tuple[float, float]
    teager_emphasis_hz: float
    teager_smoothing_seconds: float
    template_window_seconds: tuple[float, float]
    minimum_rr_seconds: float
    maximum_rr_seconds: float
    candidate_refractory_seconds: float
    candidate_prominence_mad: float
    correlation_threshold: float
    refinement_iterations: int


@dataclass(frozen=True, slots=True)
class DetectionRunConfig:
    """Input and output paths for one independent marker run."""

    input_vhdr: Path
    output_vhdr: Path
    detector: DetectorConfig


@dataclass(frozen=True, slots=True)
class CorrectionRunConfig:
    """Input, output, detector, and method settings for one BCG correction."""

    input_vhdr: Path
    output_vhdr: Path
    detector: DetectorConfig
    method: str
    window_seconds: tuple[float, float]
    ecg_to_bcg_delay_seconds: float
    aas_neighbor_count: int
    pca_obs_components: int
    evaluation: EvaluationSettings
    #: Largest share of a recording that may sit inside RR gaps. Inside a gap
    #: no template is subtracted, so that span keeps full-amplitude BCG and is
    #: marked bad in the output. Above this share the detected beats cannot be
    #: trusted either, so the recording is refused instead of half-corrected.
    maximum_gap_fraction: float


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    """Input roots and locked settings for the paired BCG benchmark."""

    fastr_root: Path
    analyzer_input_root: Path
    analyzer_output_root: Path
    output_root: Path
    detector: DetectorConfig
    marker_tolerance_seconds: float
    correction_methods: tuple[str, ...]
    correction_window_seconds: tuple[float, float]
    ecg_to_bcg_delay_seconds: float
    aas_neighbor_count: int
    pca_obs_components: int
    null_surrogate_count: int
    random_seed: int


_DETECTION_TOP_LEVEL_KEYS = frozenset({"input", "output", "detector"})
_DETECTION_INPUT_KEYS = frozenset({"vhdr"})
_DETECTION_OUTPUT_KEYS = frozenset({"vhdr"})
_CORRECTION_TOP_LEVEL_KEYS = frozenset({"input", "output", "detector", "correction"})
_CORRECTION_KEYS = frozenset(
    {
        "method",
        "window_seconds",
        "ecg_to_bcg_delay_seconds",
        "aas_neighbor_count",
        "pca_obs_components",
        "evaluation",
        "maximum_gap_fraction",
    }
)
_BENCHMARK_TOP_LEVEL_KEYS = frozenset({"benchmark", "detector"})
_BENCHMARK_KEYS = frozenset(
    {
        "fastr_root",
        "analyzer_input_root",
        "analyzer_output_root",
        "output_root",
        "marker_tolerance_seconds",
        "correction_methods",
        "correction_window_seconds",
        "ecg_to_bcg_delay_seconds",
        "aas_neighbor_count",
        "pca_obs_components",
        "null_surrogate_count",
        "random_seed",
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
#: Imported rather than restated, so a new method cannot be accepted by
#: ``correct_bcg`` while this loader still rejects it.
_SUPPORTED_CORRECTION_METHODS = METHODS


def _read_document(path: str | Path) -> tuple[Path, object]:
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
    return config_path, document


def _mapping(value: object, field_path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{field_path} must be a mapping")
    return value


def _reject_unknown_keys(
    values: Mapping[str, object],
    expected_keys: frozenset[str],
    field_path: str,
) -> None:
    unknown = sorted(str(key) for key in values if key not in expected_keys)
    if unknown:
        raise ConfigurationError(
            f"unknown field(s) in {field_path}: {', '.join(unknown)}"
        )


def _require_keys(
    values: Mapping[str, object],
    expected_keys: frozenset[str],
    field_path: str,
) -> None:
    for key in sorted(expected_keys):
        if key not in values:
            raise ConfigurationError(f"missing required field: {field_path}.{key}")


def _section(
    root: Mapping[str, object],
    name: str,
    expected_keys: frozenset[str],
) -> Mapping[str, object]:
    values = _mapping(root[name], name)
    _reject_unknown_keys(values, expected_keys, name)
    _require_keys(values, expected_keys, name)
    return values


def _required(values: Mapping[str, object], name: str) -> object:
    if name not in values:
        raise ConfigurationError(f"missing required field: {name}")
    return values[name]


def _string(values: Mapping[str, object], name: str) -> str:
    value = _required(values, name)
    if not isinstance(value, str) or not value:
        raise ConfigurationError(f"{name} must be a nonempty string")
    if any(character in value for character in ("\0", "\n", "\r")):
        raise ConfigurationError(f"{name} contains an invalid character")
    return value


def _path_value(
    values: Mapping[str, object],
    name: str,
    base_directory: Path,
) -> Path:
    value = _string(values, name)
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_directory / path
    return path.resolve()


def _number(
    values: Mapping[str, object],
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    value = _required(values, name)
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ConfigurationError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ConfigurationError(f"{name} must be a finite number")
    if minimum is not None and number <= minimum:
        raise ConfigurationError(f"{name} must be greater than {minimum}")
    if maximum is not None and number > maximum:
        raise ConfigurationError(f"{name} must be less than or equal to {maximum}")
    return number


def _positive_number(values: Mapping[str, object], name: str) -> float:
    return _number(values, name, minimum=0.0)


def _nonnegative_number(values: Mapping[str, object], name: str) -> float:
    number = _required(values, name)
    if isinstance(number, bool) or not isinstance(number, Real):
        raise ConfigurationError(f"{name} must be a finite nonnegative number")
    result = float(number)
    if not math.isfinite(result) or result < 0.0:
        raise ConfigurationError(f"{name} must be a finite nonnegative number")
    return result


def _integer(values: Mapping[str, object], name: str, *, minimum: int) -> int:
    value = _required(values, name)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigurationError(
            f"{name} must be an integer greater than or equal to {minimum}"
        )
    return value


def _even_integer(
    values: Mapping[str, object],
    name: str,
    *,
    minimum: int,
) -> int:
    value = _integer(values, name, minimum=minimum)
    if value % 2:
        raise ConfigurationError(f"{name} must be even")
    return value


def _number_value(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ConfigurationError(f"{name} must contain two finite numbers")
    number = float(value)
    if not math.isfinite(number):
        raise ConfigurationError(f"{name} must contain two finite numbers")
    return number


def _interval(
    values: Mapping[str, object],
    name: str,
) -> tuple[float, float]:
    value = _required(values, name)
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ConfigurationError(f"{name} must contain two finite numbers")
    if len(value) != 2:
        raise ConfigurationError(f"{name} must contain exactly two numbers")
    return (
        _number_value(value[0], name),
        _number_value(value[1], name),
    )


def _ordered_interval(
    values: Mapping[str, object],
    name: str,
) -> tuple[float, float]:
    interval = _interval(values, name)
    if interval[0] >= interval[1]:
        raise ConfigurationError(f"{name} must be increasing")
    return interval


def _ordered_positive_interval(
    values: Mapping[str, object],
    name: str,
) -> tuple[float, float]:
    interval = _ordered_interval(values, name)
    if interval[0] <= 0.0:
        raise ConfigurationError(f"{name} must contain positive values")
    return interval


def _correction_methods(values: Mapping[str, object]) -> tuple[str, ...]:
    value = _required(values, "correction_methods")
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ConfigurationError("correction_methods must be a nonempty list")
    if not value:
        raise ConfigurationError("correction_methods must be a nonempty list")
    if any(not isinstance(method, str) for method in value):
        raise ConfigurationError("correction_methods must contain strings")
    methods = tuple(value)
    unsupported = sorted(set(methods) - _SUPPORTED_CORRECTION_METHODS)
    if unsupported:
        supported = ", ".join(sorted(_SUPPORTED_CORRECTION_METHODS))
        raise ConfigurationError(
            f"correction_methods must contain only {supported}; "
            f"got {', '.join(unsupported)}"
        )
    if len(set(methods)) != len(methods):
        raise ConfigurationError("correction_methods must not contain duplicates")
    return methods


def _detector_config(values: Mapping[str, object]) -> DetectorConfig:
    preprocessing_band = _ordered_positive_interval(
        values,
        "preprocessing_band_hz",
    )
    template_window = _ordered_interval(values, "template_window_seconds")
    minimum_rr = _positive_number(values, "minimum_rr_seconds")
    maximum_rr = _positive_number(values, "maximum_rr_seconds")
    if minimum_rr >= maximum_rr:
        raise ConfigurationError(
            "minimum_rr_seconds must be less than maximum_rr_seconds"
        )

    correlation_threshold = _number(
        values,
        "correlation_threshold",
        minimum=0.0,
        maximum=1.0,
    )
    return DetectorConfig(
        ecg_channel=_string(values, "ecg_channel"),
        preprocessing_band_hz=preprocessing_band,
        teager_emphasis_hz=_positive_number(values, "teager_emphasis_hz"),
        teager_smoothing_seconds=_positive_number(
            values,
            "teager_smoothing_seconds",
        ),
        template_window_seconds=template_window,
        minimum_rr_seconds=minimum_rr,
        maximum_rr_seconds=maximum_rr,
        candidate_refractory_seconds=_positive_number(
            values,
            "candidate_refractory_seconds",
        ),
        candidate_prominence_mad=_positive_number(
            values,
            "candidate_prominence_mad",
        ),
        correlation_threshold=correlation_threshold,
        refinement_iterations=_integer(
            values,
            "refinement_iterations",
            minimum=1,
        ),
    )


def load_detection_config(path: str | Path) -> DetectionRunConfig:
    config_path, document = _read_document(path)
    root = _mapping(document, "configuration")
    _reject_unknown_keys(root, _DETECTION_TOP_LEVEL_KEYS, "configuration")
    _require_keys(root, _DETECTION_TOP_LEVEL_KEYS, "configuration")

    input_values = _section(root, "input", _DETECTION_INPUT_KEYS)
    output_values = _section(root, "output", _DETECTION_OUTPUT_KEYS)
    detector_values = _section(root, "detector", _DETECTOR_KEYS)
    base_directory = config_path.parent
    return DetectionRunConfig(
        input_vhdr=_path_value(input_values, "vhdr", base_directory),
        output_vhdr=_path_value(output_values, "vhdr", base_directory),
        detector=_detector_config(detector_values),
    )


def load_correction_config(path: str | Path) -> CorrectionRunConfig:
    config_path, document = _read_document(path)
    root = _mapping(document, "configuration")
    _reject_unknown_keys(root, _CORRECTION_TOP_LEVEL_KEYS, "configuration")
    _require_keys(root, _CORRECTION_TOP_LEVEL_KEYS, "configuration")

    input_values = _section(root, "input", _DETECTION_INPUT_KEYS)
    output_values = _section(root, "output", _DETECTION_OUTPUT_KEYS)
    detector_values = _section(root, "detector", _DETECTOR_KEYS)
    correction_values = _section(root, "correction", _CORRECTION_KEYS)
    method = _string(correction_values, "method")
    if method not in _SUPPORTED_CORRECTION_METHODS:
        supported = ", ".join(sorted(_SUPPORTED_CORRECTION_METHODS))
        raise ConfigurationError(
            f"correction.method must be one of {supported}; got {method!r}"
        )
    base_directory = config_path.parent
    return CorrectionRunConfig(
        input_vhdr=_path_value(input_values, "vhdr", base_directory),
        output_vhdr=_path_value(output_values, "vhdr", base_directory),
        detector=_detector_config(detector_values),
        method=method,
        window_seconds=_ordered_interval(correction_values, "window_seconds"),
        ecg_to_bcg_delay_seconds=_nonnegative_number(
            correction_values,
            "ecg_to_bcg_delay_seconds",
        ),
        aas_neighbor_count=_even_integer(
            correction_values,
            "aas_neighbor_count",
            minimum=2,
        ),
        pca_obs_components=_integer(
            correction_values,
            "pca_obs_components",
            minimum=1,
        ),
        evaluation=parse_evaluation(correction_values["evaluation"]),
        maximum_gap_fraction=_number(
            correction_values,
            "maximum_gap_fraction",
            minimum=0.0,
            maximum=1.0,
        ),
    )


def load_benchmark_config(path: str | Path) -> BenchmarkConfig:
    config_path, document = _read_document(path)
    root = _mapping(document, "configuration")
    _reject_unknown_keys(root, _BENCHMARK_TOP_LEVEL_KEYS, "configuration")
    _require_keys(root, _BENCHMARK_TOP_LEVEL_KEYS, "configuration")

    benchmark_values = _section(root, "benchmark", _BENCHMARK_KEYS)
    detector_values = _section(root, "detector", _DETECTOR_KEYS)
    base_directory = config_path.parent
    return BenchmarkConfig(
        fastr_root=_path_value(benchmark_values, "fastr_root", base_directory),
        analyzer_input_root=_path_value(
            benchmark_values,
            "analyzer_input_root",
            base_directory,
        ),
        analyzer_output_root=_path_value(
            benchmark_values,
            "analyzer_output_root",
            base_directory,
        ),
        output_root=_path_value(benchmark_values, "output_root", base_directory),
        detector=_detector_config(detector_values),
        marker_tolerance_seconds=_positive_number(
            benchmark_values,
            "marker_tolerance_seconds",
        ),
        correction_methods=_correction_methods(benchmark_values),
        correction_window_seconds=_ordered_interval(
            benchmark_values,
            "correction_window_seconds",
        ),
        ecg_to_bcg_delay_seconds=_nonnegative_number(
            benchmark_values,
            "ecg_to_bcg_delay_seconds",
        ),
        aas_neighbor_count=_even_integer(
            benchmark_values,
            "aas_neighbor_count",
            minimum=2,
        ),
        pca_obs_components=_integer(
            benchmark_values,
            "pca_obs_components",
            minimum=1,
        ),
        null_surrogate_count=_integer(
            benchmark_values,
            "null_surrogate_count",
            minimum=1,
        ),
        random_seed=_integer(benchmark_values, "random_seed", minimum=0),
    )
