"""Held-out cohort validation of blocked-mean BCG correction."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from numbers import Integral
from pathlib import Path

import mne
import numpy as np
import yaml

from bcg_correction.bcg import BcgCorrectionConfig, correct_bcg, rr_gap_spans
from bcg_correction.cardiac import detect_r_peaks
from bcg_correction.correction_report import (
    CorrectionProfile,
    compute_correction_profile,
    save_topography_report,
    write_profile,
)
from bcg_correction.metrics import delay_estimation_eeg, estimate_ecg_to_bcg_delay
from bcgnet.compare.arms import AAS, BCGNET, BLOCKED_MEAN, PCA_OBS
from bcgnet.compare.config import load_compare_config
from bcgnet.compare.pairs import pair_recordings

_CONFIG_KEYS = frozenset(
    {
        "compare_config",
        "output_root",
        "recordings",
        "cross_fit_fold_count",
    }
)
_PROFILE_METRICS = (
    "locked_ratio",
    "locked_ratio_raw",
    "locked_before_uv",
    "locked_after_uv",
    "specificity",
    "alpha_collateral_fraction",
    "beats",
    "heart_rate_bpm",
    "applied_delay_seconds",
    "gap_fraction",
)


@dataclass(frozen=True, slots=True)
class HoldoutRecording:
    """One explicitly requested recording."""

    subject: str
    label: str


@dataclass(frozen=True, slots=True)
class HoldoutConfig:
    """Strict settings for one untouched-cohort validation."""

    compare_config: Path
    output_root: Path
    recordings: tuple[HoldoutRecording, ...]
    cross_fit_fold_count: int


def _load_config(path: Path) -> HoldoutConfig:
    config_path = path.expanduser().resolve()
    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("holdout configuration must be a mapping")
    keys = frozenset(document)
    unknown = sorted(keys - _CONFIG_KEYS)
    if unknown:
        raise ValueError(
            "holdout configuration has unknown keys: " + ", ".join(unknown)
        )
    missing = sorted(_CONFIG_KEYS - keys)
    if missing:
        raise ValueError("holdout configuration is missing keys: " + ", ".join(missing))
    base = config_path.parent
    return HoldoutConfig(
        compare_config=_resolve_path(
            document["compare_config"], base, "compare_config"
        ),
        output_root=_resolve_path(document["output_root"], base, "output_root"),
        recordings=_recording_requests(document["recordings"]),
        cross_fit_fold_count=_positive_integer(
            document["cross_fit_fold_count"],
            name="cross_fit_fold_count",
        ),
    )


def _resolve_path(value: object, base: Path, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty path string")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _recording_requests(value: object) -> tuple[HoldoutRecording, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("recordings must be a nonempty list")
    requests = []
    for index, item in enumerate(value):
        if not isinstance(item, dict) or frozenset(item) != {"subject", "label"}:
            raise ValueError(
                f"recordings[{index}] must contain only subject and label"
            )
        subject = item["subject"]
        label = item["label"]
        if not all(
            isinstance(text, str) and text and text == text.strip()
            for text in (subject, label)
        ):
            raise ValueError(
                f"recordings[{index}] subject and label must be nonempty strings"
            )
        requests.append(HoldoutRecording(subject=subject, label=label))
    pairs = [(item.subject, item.label) for item in requests]
    if len(set(pairs)) != len(pairs):
        raise ValueError("recordings must contain unique subject-label pairs")
    return tuple(requests)


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)



def _correct_bounded(
    data: np.ndarray,
    channel_names: tuple[str, ...],
    *,
    ecg_index: int,
    peak_samples: np.ndarray,
    sampling_rate: float,
    delay_seconds: float,
    comparison,
    method: str,
    fold_count: int | None = None,
) -> np.ndarray:
    eeg_picks = np.asarray(
        [index for index in range(data.shape[0]) if index != ecg_index],
        dtype=np.int64,
    )
    return correct_bcg(
        data,
        peak_samples,
        sampling_rate,
        channel_names=channel_names,
        eeg_picks=eeg_picks,
        ecg_channel_index=ecg_index,
        config=BcgCorrectionConfig(
            method=method,
            window_seconds=comparison.correction.window_seconds,
            ecg_to_bcg_delay_seconds=delay_seconds,
            aas_neighbor_count=comparison.correction.aas_neighbor_count,
            pca_obs_components=comparison.correction.pca_obs_components,
            cross_fit_fold_count=(
                comparison.correction.cross_fit_fold_count
                if fold_count is None
                else fold_count
            ),
        ),
    ).data_volts


def _read_recording(path: Path) -> tuple[np.ndarray, tuple[str, ...], float]:
    raw = mne.io.read_raw_brainvision(path, preload=False, verbose="ERROR")
    try:
        return raw.get_data(), tuple(raw.ch_names), float(raw.info["sfreq"])
    finally:
        raw.close()


def _read_aligned_correction(
    path: Path,
    channel_names: tuple[str, ...],
    sample_count: int,
    sampling_rate: float,
) -> np.ndarray:
    data, corrected_names, corrected_rate = _read_recording(path)
    if corrected_names != channel_names:
        raise RuntimeError(f"corrected channel order differs from FASTR: {path}")
    if data.shape[1] != sample_count or corrected_rate != sampling_rate:
        raise RuntimeError(f"corrected sample grid differs from FASTR: {path}")
    return data


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError("refusing to write an empty holdout table")
    fields = tuple(rows[0])
    with path.open("x", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: object) -> None:
    with path.open("x", encoding="utf-8") as output:
        json.dump(payload, output, indent=2, default=str)
        output.write("\n")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    holdout = _load_config(arguments.config)
    comparison = load_compare_config(holdout.compare_config)

    available = {
        (recording.bids_id, recording.label): recording
        for recording in pair_recordings(comparison)
    }
    requested = [(item.subject, item.label) for item in holdout.recordings]
    missing = [key for key in requested if key not in available]
    if missing:
        raise RuntimeError(f"requested recordings do not exist: {missing}")
    selected = [available[key] for key in requested]
    missing_bcgnet = [
        (recording.bids_id, recording.label)
        for recording in selected
        if not recording.has(BCGNET)
    ]
    if missing_bcgnet:
        raise RuntimeError(
            f"requested BCGNet corrections do not exist: {missing_bcgnet}"
        )
    holdout.output_root.mkdir(parents=True, exist_ok=False)

    rows: list[dict[str, object]] = []
    groups: dict[str, list[CorrectionProfile]] = {
        BLOCKED_MEAN.label: [],
        AAS.label: [],
        PCA_OBS.label: [],
        BCGNET.label: [],
    }
    for recording in selected:
        raw_data, channel_names, sampling_rate = _read_recording(recording.fastr_vhdr)
        ecg_index = channel_names.index("ECG")
        detection = detect_r_peaks(
            raw_data[ecg_index],
            sampling_rate,
            config=comparison.correction.detector,
        )
        peak_samples = detection.peak_samples
        delay_seconds = estimate_ecg_to_bcg_delay(
            delay_estimation_eeg(
                raw_data,
                channel_names,
                ecg_channel_index=ecg_index,
            ),
            peak_samples,
            sampling_rate_hz=sampling_rate,
        ).best_delay_seconds
        gaps = rr_gap_spans(
            peak_samples,
            sampling_rate,
            comparison.correction.detector.maximum_rr_seconds,
        )
        gap_fraction = sum(stop - start for start, stop in gaps) / raw_data.shape[1]
        # Every bounded arm goes through ``correct_bcg`` so the tool cannot
        # drift from what ``bcg blocked-mean`` writes.
        blocked = _correct_bounded(
            raw_data,
            channel_names,
            ecg_index=ecg_index,
            peak_samples=peak_samples,
            sampling_rate=sampling_rate,
            delay_seconds=delay_seconds,
            comparison=comparison,
            method=BLOCKED_MEAN.key,
            fold_count=holdout.cross_fit_fold_count,
        )
        corrections = (
            (BLOCKED_MEAN.key, BLOCKED_MEAN.label, blocked),
            (
                AAS.key,
                AAS.label,
                _correct_bounded(
                    raw_data,
                    channel_names,
                    ecg_index=ecg_index,
                    peak_samples=peak_samples,
                    sampling_rate=sampling_rate,
                    delay_seconds=delay_seconds,
                    comparison=comparison,
                    method=AAS.key,
                ),
            ),
            (
                PCA_OBS.key,
                PCA_OBS.label,
                _correct_bounded(
                    raw_data,
                    channel_names,
                    ecg_index=ecg_index,
                    peak_samples=peak_samples,
                    sampling_rate=sampling_rate,
                    delay_seconds=delay_seconds,
                    comparison=comparison,
                    method=PCA_OBS.key,
                ),
            ),
            (
                BCGNET.key,
                BCGNET.label,
                _read_aligned_correction(
                    recording.cleaned_vhdr[BCGNET.key],
                    channel_names,
                    raw_data.shape[1],
                    sampling_rate,
                ),
            ),
        )
        for method, label, corrected in corrections:
            profile = compute_correction_profile(
                raw_data,
                corrected,
                channel_names,
                ecg_channel_index=ecg_index,
                peak_samples=peak_samples,
                sampling_rate_hz=sampling_rate,
                delay_seconds=delay_seconds,
                window_seconds=comparison.correction.window_seconds,
                gap_fraction=gap_fraction,
                method=method,
                label=recording.label,
            )
            if profile is None:
                raise RuntimeError(f"too few beats for {recording.stem} {method}")
            groups[label].append(profile)
            profile_path = (
                holdout.output_root
                / "profiles"
                / recording.bids_id
                / f"{recording.stem}_{method}_profile.npz"
            )
            write_profile(profile, profile_path)
            rows.append(
                {
                    "subject": recording.bids_id,
                    "recording": recording.label,
                    "input_vhdr": str(recording.fastr_vhdr),
                    "method": method,
                    "detector_status": detection.quality.status,
                    **{
                        metric: getattr(profile, metric)
                        for metric in _PROFILE_METRICS
                    },
                }
            )
        print(f"{recording.bids_id} {recording.label}: 4 methods", flush=True)

    _write_csv(holdout.output_root / "holdout_metrics.csv", rows)
    _write_json(
        holdout.output_root / "run.json",
        {
            **asdict(holdout),
            "recording_count": len(selected),
            "methods": list(groups),
        },
    )
    saved = save_topography_report(
        groups,
        title=(
            "Held-out cohort — where each BCG correction acts, "
            f"{len(selected)} recordings, "
            f"{len({item.bids_id for item in selected})} participants"
        ),
        output=holdout.output_root / "cohort_topography.png",
    )
    if not saved:
        raise RuntimeError("topography could not be rendered")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
