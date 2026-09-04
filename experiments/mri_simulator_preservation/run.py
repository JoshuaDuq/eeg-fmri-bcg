#!/usr/bin/env python3
# ruff: noqa: E402
"""Paired simulator vs FASTR vs scanner-corrected thermal 8-13 Hz response."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from bcgnet.compare.arms import CLEAN_ARMS
from bcgstudy.simulator_experiment import (
    discover_simulator_recordings,
    rewrite_header_sidecars,
)
from bcgstudy.thermal_response import (
    CORRECTED_METHODS,
    METHOD_ORDER,
    SCANNER_SERIES,
    ThermalProtocol,
    ThermalRecording,
    add_fastr_improvement,
    best_trial_summary,
    cardiac_residual_from_summary,
    choose_completed_thermal,
    count_trig_therm,
    epoch_band_power,
    epoch_spectrum,
    extract_windows,
    median_spectrum,
    participant_method_summaries,
    plot_prestim_plateau_spectra,
    plot_response_spectra,
    posterior_indices,
    preprocess_raw,
    read_trial_sequence,
    recording_duration_seconds,
    reference_keep_mask,
    response_ratio_db,
    response_spectrum_db,
    sequences_match,
    trial_keep_mask,
    trig_therm_samples,
)

_RUN = re.compile(r"(?<![0-9])run[ _.-]?(\d+)(?![0-9])", re.IGNORECASE)


def _load_config(path: Path) -> dict:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("config must be a mapping")
    return document


def _stage_simulator(vhdr: Path, work_root: Path, bids_id: str) -> Path:
    staged_dir = work_root / "staged" / bids_id
    staged_dir.mkdir(parents=True, exist_ok=True)
    staged = staged_dir / vhdr.name
    if not staged.exists():
        rewrite_header_sidecars(vhdr, staged)
        for suffix in (".eeg", ".vmrk"):
            target = staged.with_suffix(suffix)
            if not target.exists():
                os.symlink(vhdr.with_suffix(suffix), target)
    return staged


def _run_from_name(name: str) -> int | None:
    match = _RUN.search(name)
    return int(match.group(1)) if match else None


def _find_thermal_vhdr(
    folder: Path, run: int, *, suffix: str | None = None
) -> Path | None:
    if not folder.is_dir():
        return None
    hits = []
    for path in folder.glob("*.vhdr"):
        if path.name.startswith("._"):
            continue
        if "thermal" not in path.name.lower():
            continue
        if _run_from_name(path.stem) != run:
            continue
        if suffix is not None and not path.stem.endswith(f"_{suffix}"):
            continue
        hits.append(path)
    if len(hits) > 1:
        names = ", ".join(path.name for path in hits)
        raise ValueError(f"more than one thermal VHDR for run {run}: {names}")
    if not hits:
        return None
    return hits[0]


def _completed_simulator(
    recordings, bids_id: str, run: int, *, expected_triggers: int
) -> ThermalRecording | None:
    candidates = []
    for recording in recordings:
        if recording.bids_id != bids_id or recording.run != run:
            continue
        vmrk = recording.source_vhdr.with_suffix(".vmrk")
        candidates.append(
            ThermalRecording(
                bids_id=bids_id,
                run=run,
                name=recording.source_vhdr.stem,
                vhdr=recording.source_vhdr,
                n_trig_therm=count_trig_therm(vmrk) if vmrk.is_file() else 0,
                duration_seconds=recording_duration_seconds(recording.source_vhdr),
            )
        )
    return choose_completed_thermal(candidates, expected_triggers=expected_triggers)


def _load_compare_summary(
    path: Path,
    *,
    cardiac_template: str,
) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"missing comparison summary: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "bids_id",
            "run",
            *(cardiac_template.format(method=arm.key) for arm in CLEAN_ARMS),
        }
        missing = sorted(required - set(reader.fieldnames or ()))
        if missing:
            raise ValueError(
                f"comparison summary is missing columns: {', '.join(missing)}"
            )
        return list(reader)


def _validate_scanner_alignment(
    fastr_raw,
    fastr_events: np.ndarray,
    candidate_raw,
    candidate_events: np.ndarray,
    label: str,
) -> None:
    if tuple(candidate_raw.ch_names) != tuple(fastr_raw.ch_names):
        raise ValueError(f"{label} channel order differs from FASTR")
    if not np.isclose(candidate_raw.info["sfreq"], fastr_raw.info["sfreq"]):
        raise ValueError(f"{label} sampling rate differs from FASTR")
    if candidate_raw.n_times != fastr_raw.n_times:
        raise ValueError(f"{label} sample count differs from FASTR")
    if not np.array_equal(candidate_events, fastr_events):
        raise ValueError(f"{label} event samples differ from FASTR")


def _read_raw_and_events(vhdr: Path):
    import mne

    raw = mne.io.read_raw_brainvision(vhdr, preload=True, verbose="ERROR")
    events = np.asarray(trig_therm_samples(vhdr.with_suffix(".vmrk")), dtype=int)
    return raw, events


def _preprocess_with_events(
    raw,
    event_samples: np.ndarray,
    *,
    target_hz: float,
    filter_hz: tuple[float, float],
):
    raw = preprocess_raw(raw, filter_hz=filter_hz)
    event_matrix = np.column_stack(
        (
            np.asarray(event_samples, dtype=int),
            np.zeros(event_samples.size, dtype=int),
            np.ones(event_samples.size, dtype=int),
        )
    )
    if not np.isclose(raw.info["sfreq"], target_hz):
        raw, event_matrix = raw.resample(
            target_hz,
            events=event_matrix,
            npad="auto",
            verbose="ERROR",
        )
    return raw, event_matrix[:, 0]


def _measure_arm(raw, events: np.ndarray, protocol: ThermalProtocol) -> dict:
    picks = posterior_indices(raw.ch_names, protocol.posterior_channels)
    prestim, keep_pre = extract_windows(
        raw, events, protocol.prestimulus_seconds, picks=picks
    )
    plateau, keep_plat = extract_windows(
        raw, events, protocol.plateau_seconds, picks=picks
    )
    sfreq = float(raw.info["sfreq"])
    p_pre = epoch_band_power(
        prestim,
        sfreq=sfreq,
        band=protocol.band_hz,
        nperseg_seconds=protocol.welch_nperseg_seconds,
    )
    p_plat = epoch_band_power(
        plateau,
        sfreq=sfreq,
        band=protocol.band_hz,
        nperseg_seconds=protocol.welch_nperseg_seconds,
    )
    freq, psd_pre = epoch_spectrum(
        prestim,
        sfreq=sfreq,
        nperseg_seconds=protocol.welch_nperseg_seconds,
        fmin=1.0,
        fmax=40.0,
    )
    _, psd_plat = epoch_spectrum(
        plateau,
        sfreq=sfreq,
        nperseg_seconds=protocol.welch_nperseg_seconds,
        fmin=1.0,
        fmax=40.0,
    )
    return {
        "complete": keep_pre & keep_plat,
        "amplitude_quality": (
            trial_keep_mask(prestim, peak_to_peak_uv=protocol.peak_to_peak_uv)
            & trial_keep_mask(plateau, peak_to_peak_uv=protocol.peak_to_peak_uv)
        ),
        "p_pre": p_pre,
        "p_plat": p_plat,
        "freq": freq,
        "psd_pre": psd_pre,
        "psd_plat": psd_plat,
        "r": np.array(
            [
                response_ratio_db(float(a), float(b))
                for a, b in zip(p_plat, p_pre, strict=True)
            ]
        ),
    }


def _exclusion_row(
    bids_id: str,
    run: int,
    reason: str,
    missing_methods: str = "",
) -> dict:
    return {
        "bids_id": bids_id,
        "run": run,
        "status": "excluded",
        "reason": reason,
        "missing_methods": missing_methods,
        "n_triggers": "",
        "n_kept": "",
        "n_dropped": "",
    }


def _pair_runs(
    config: dict, protocol: ThermalProtocol
) -> tuple[list[dict], list[dict]]:
    paths = config["paths"]
    exclude = set(config["subjects"]["exclude"])
    include = [
        bids_id
        for bids_id in config["subjects"]["include"]
        if bids_id not in exclude
    ]
    simulator_recordings = discover_simulator_recordings(Path(paths["simulator_root"]))
    pairs = []
    inclusion_rows = []
    for bids_id in include:
        sim_folder = (
            Path(paths["simulator_root"]) / bids_id.replace("-", "_") / "PsychoPy_Data"
        )
        scan_folder = Path(paths["scanner_psychopy_root"]) / bids_id / "PsychoPy_Data"
        for run in protocol.runs:
            sim_rec = _completed_simulator(
                simulator_recordings,
                bids_id,
                run,
                expected_triggers=protocol.n_triggers,
            )
            if sim_rec is None:
                inclusion_rows.append(
                    _exclusion_row(bids_id, run, "no_completed_simulator")
                )
                continue
            sim_summary = best_trial_summary(
                sim_folder, run, expected_trials=protocol.n_triggers
            )
            scan_summary = best_trial_summary(
                scan_folder, run, expected_trials=protocol.n_triggers
            )
            if sim_summary is None or scan_summary is None:
                inclusion_rows.append(
                    _exclusion_row(bids_id, run, "missing_trial_summary")
                )
                continue
            sim_seq = read_trial_sequence(sim_summary)
            scan_seq = read_trial_sequence(scan_summary)
            if not sequences_match(sim_seq, scan_seq):
                inclusion_rows.append(_exclusion_row(bids_id, run, "sequence_mismatch"))
                continue
            scanner = {
                "fastr": _find_thermal_vhdr(Path(paths["fastr_root"]) / bids_id, run),
                **{
                    arm.key: _find_thermal_vhdr(
                        Path(paths[f"{arm.key}_root"]) / bids_id,
                        run,
                        suffix=arm.suffix,
                    )
                    for arm in CLEAN_ARMS
                },
            }
            if scanner["fastr"] is None:
                inclusion_rows.append(
                    _exclusion_row(bids_id, run, "missing_fastr", "fastr")
                )
                continue
            missing = [key for key in CORRECTED_METHODS if scanner[key] is None]
            if missing:
                inclusion_rows.append(
                    _exclusion_row(
                        bids_id, run, "missing_corrected_arm", ";".join(missing)
                    )
                )
                continue
            pairs.append(
                {
                    "bids_id": bids_id,
                    "run": run,
                    "sequence": sim_seq,
                    "simulator": sim_rec.vhdr,
                    **scanner,
                }
            )
            inclusion_rows.append(
                {
                    "bids_id": bids_id,
                    "run": run,
                    "status": "included",
                    "reason": "",
                    "missing_methods": "",
                    "n_triggers": "",
                    "n_kept": "",
                    "n_dropped": "",
                }
            )
    return pairs, inclusion_rows


def _update_inclusion(
    rows: list[dict],
    bids_id: str,
    run: int,
    *,
    n_triggers: int,
    n_kept: int,
    n_dropped: int,
) -> None:
    for row in rows:
        if row["bids_id"] == bids_id and int(row["run"]) == run:
            row["n_triggers"] = n_triggers
            row["n_kept"] = n_kept
            row["n_dropped"] = n_dropped
            return


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("config.yaml"),
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="measure at most N paired runs"
    )
    arguments = parser.parse_args()
    config = _load_config(arguments.config.resolve())
    protocol = ThermalProtocol.from_mapping(config["thermal"])
    paths = config["paths"]
    output_root = Path(paths["output_root"])
    if not output_root.is_absolute():
        output_root = Path(__file__).resolve().parent / output_root
    work_root = Path(paths["work_root"])
    target_hz = float(config["preprocess"]["target_hz"])
    filter_hz = (
        float(config["preprocess"]["filter_hz"][0]),
        float(config["preprocess"]["filter_hz"][1]),
    )
    cardiac_template = str(config["cardiac"]["column_template"])
    compare_rows = _load_compare_summary(
        Path(paths["compare_summary"]), cardiac_template=cardiac_template
    )

    pairs, inclusion_rows = _pair_runs(config, protocol)
    if arguments.limit:
        pairs = pairs[: arguments.limit]
    print(f"paired runs: {len(pairs)}")

    trial_rows: list[dict] = []
    spectra_r: dict[str, dict[str, list[np.ndarray]]] = defaultdict(
        lambda: defaultdict(list)
    )
    spectra_psd: dict[str, dict[str, list[tuple[np.ndarray, np.ndarray]]]] = (
        defaultdict(lambda: defaultdict(list))
    )
    freq_ref = None
    runs_by_subject: dict[str, list[int]] = defaultdict(list)

    for pair in pairs:
        bids_id = pair["bids_id"]
        run = pair["run"]
        print(f"measure {bids_id} run{run}")
        sim_vhdr = _stage_simulator(pair["simulator"], work_root, bids_id)
        sim_raw, sim_events = _read_raw_and_events(sim_vhdr)
        if sim_events.size != protocol.n_triggers:
            raise ValueError(
                f"{bids_id} run{run} simulator Trig_therm={sim_events.size}"
            )
        sim_raw, sim_events = _preprocess_with_events(
            sim_raw, sim_events, target_hz=target_hz, filter_hz=filter_hz
        )
        fastr_raw, fastr_events = _read_raw_and_events(pair["fastr"])
        if fastr_events.size != protocol.n_triggers:
            raise ValueError(f"{bids_id} run{run} FASTR Trig_therm={fastr_events.size}")
        corrected_raws = {}
        for method in CORRECTED_METHODS:
            arm_raw, arm_events = _read_raw_and_events(pair[method])
            _validate_scanner_alignment(
                fastr_raw, fastr_events, arm_raw, arm_events, method
            )
            corrected_raws[method] = arm_raw
        fastr_raw, fastr_events = _preprocess_with_events(
            fastr_raw, fastr_events, target_hz=target_hz, filter_hz=filter_hz
        )
        measured = {
            "simulator": _measure_arm(sim_raw, sim_events, protocol),
            "fastr": _measure_arm(fastr_raw, fastr_events, protocol),
        }
        del sim_raw, fastr_raw
        for method, arm_raw in corrected_raws.items():
            arm_raw, _events = _preprocess_with_events(
                arm_raw, fastr_events, target_hz=target_hz, filter_hz=filter_hz
            )
            measured[method] = _measure_arm(arm_raw, fastr_events, protocol)
            del arm_raw
        keep = reference_keep_mask(
            measured["simulator"]["complete"],
            measured["simulator"]["amplitude_quality"],
            measured["fastr"]["complete"],
        )
        for method in CORRECTED_METHODS:
            values = measured[method]["r"][keep]
            if values.size and not np.all(np.isfinite(values)):
                raise ValueError(
                    f"{bids_id} run{run} {method} has non-finite R on included trials"
                )
        n_keep = int(np.count_nonzero(keep))
        n_drop = int(np.count_nonzero(~keep))
        print(f"  trials kept {n_keep}/{keep.size} (dropped {n_drop})")
        _update_inclusion(
            inclusion_rows,
            bids_id,
            run,
            n_triggers=int(keep.size),
            n_kept=n_keep,
            n_dropped=n_drop,
        )
        if n_keep == 0:
            continue
        runs_by_subject[bids_id].append(run)
        sim = measured["simulator"]
        freq_ref = sim["freq"] if freq_ref is None else freq_ref
        uv = 1e12
        for key in METHOD_ORDER:
            if key not in measured:
                continue
            spectra_psd[key][bids_id].append(
                (
                    median_spectrum(measured[key]["psd_pre"], keep) * uv,
                    median_spectrum(measured[key]["psd_plat"], keep) * uv,
                )
            )
            spectra_r[key][bids_id].append(
                response_spectrum_db(
                    median_spectrum(measured[key]["psd_plat"], keep),
                    median_spectrum(measured[key]["psd_pre"], keep),
                )
            )
        for trial, kept in enumerate(keep, start=1):
            if not kept:
                continue
            for method in SCANNER_SERIES:
                trial_rows.append(
                    {
                        "bids_id": bids_id,
                        "run": run,
                        "trial": trial,
                        "method": method,
                        "stimulus_temp": pair["sequence"][trial - 1][0],
                        "surface": pair["sequence"][trial - 1][1],
                        "r_method": float(measured[method]["r"][trial - 1]),
                        "r_simulator": float(sim["r"][trial - 1]),
                        "p_plateau_method": float(
                            measured[method]["p_plat"][trial - 1]
                        ),
                        "p_prestim_method": float(measured[method]["p_pre"][trial - 1]),
                        "p_plateau_simulator": float(sim["p_plat"][trial - 1]),
                        "p_prestim_simulator": float(sim["p_pre"][trial - 1]),
                    }
                )

    participant_rows = add_fastr_improvement(participant_method_summaries(trial_rows))
    method_rank = {method: index for index, method in enumerate(SCANNER_SERIES)}
    participant_rows.sort(
        key=lambda row: (row["bids_id"], method_rank.get(row["method"], 99))
    )
    for row in participant_rows:
        row["n_runs"] = len(runs_by_subject.get(row["bids_id"], []))
        if row["method"] == "fastr":
            row["cardiac_residual"] = 1.0
        else:
            row["cardiac_residual"] = cardiac_residual_from_summary(
                compare_rows,
                bids_id=row["bids_id"],
                method=row["method"],
                runs=runs_by_subject[row["bids_id"]],
                column=cardiac_template.format(method=row["method"]),
            )

    method_rows = []
    by_method: dict[str, list[dict]] = defaultdict(list)
    for row in participant_rows:
        by_method[row["method"]].append(row)
    for method in SCANNER_SERIES:
        values = by_method.get(method, [])
        errors = np.array(
            [row["median_absolute_error"] for row in values], dtype=float
        )
        bias = np.array(
            [row["signed_median_difference"] for row in values], dtype=float
        )
        residual = np.array([row["cardiac_residual"] for row in values], dtype=float)
        improve = np.array(
            [row["improvement_vs_fastr"] for row in values], dtype=float
        )
        method_rows.append(
            {
                "method": method,
                "n_participants": len(values),
                "median_E": float(np.median(errors)) if values else float("nan"),
                "median_signed_difference": (
                    float(np.median(bias)) if values else float("nan")
                ),
                "median_improvement_vs_fastr": (
                    float(np.median(improve)) if values else float("nan")
                ),
                "median_cardiac_residual": float(np.median(residual))
                if values
                else float("nan"),
            }
        )

    _write_csv(
        output_root / "trial_pairs.csv",
        trial_rows,
        [
            "bids_id",
            "run",
            "trial",
            "method",
            "stimulus_temp",
            "surface",
            "r_method",
            "r_simulator",
            "p_plateau_method",
            "p_prestim_method",
            "p_plateau_simulator",
            "p_prestim_simulator",
        ],
    )
    _write_csv(
        output_root / "participant_summary.csv",
        participant_rows,
        [
            "bids_id",
            "method",
            "n_runs",
            "n_trials",
            "median_absolute_error",
            "signed_median_difference",
            "improvement_vs_fastr",
            "median_r_method",
            "median_r_simulator",
            "cardiac_residual",
        ],
    )
    _write_csv(
        output_root / "method_summary.csv",
        method_rows,
        [
            "method",
            "n_participants",
            "median_E",
            "median_signed_difference",
            "median_improvement_vs_fastr",
            "median_cardiac_residual",
        ],
    )
    _write_csv(
        output_root / "run_inclusion.csv",
        inclusion_rows,
        [
            "bids_id",
            "run",
            "status",
            "reason",
            "missing_methods",
            "n_triggers",
            "n_kept",
            "n_dropped",
        ],
    )

    def _stack_participant(store):
        curves = {}
        if freq_ref is None:
            return curves
        for key, by_subject in store.items():
            participant_curves = []
            for series in by_subject.values():
                stacked = np.stack(series)
                participant_curves.append(np.nanmedian(stacked, axis=0))
            stack = np.stack(participant_curves)
            median = np.nanmedian(stack, axis=0)
            q1 = q3 = None
            if stack.shape[0] >= 3:
                q1, q3 = np.nanpercentile(stack, [25, 75], axis=0)
            curves[key] = {"freq": freq_ref, "median": median, "q1": q1, "q3": q3}
        return curves

    figures = output_root / "figures"
    r_curves = _stack_participant(spectra_r)
    if r_curves:
        import matplotlib.pyplot as plt

        fig = plot_response_spectra(
            r_curves, figures / "thermal_response_spectra.png", protocol
        )
        plt.close(fig)
        psd_panels = {}
        for key, by_subject in spectra_psd.items():
            prestim = []
            plateau = []
            for series in by_subject.values():
                pre = np.nanmedian(np.stack([item[0] for item in series]), axis=0)
                plat = np.nanmedian(np.stack([item[1] for item in series]), axis=0)
                prestim.append(pre)
                plateau.append(plat)
            psd_panels[key] = {
                "freq": freq_ref,
                "prestim": np.nanmedian(np.stack(prestim), axis=0),
                "plateau": np.nanmedian(np.stack(plateau), axis=0),
            }
        fig = plot_prestim_plateau_spectra(
            psd_panels, figures / "prestim_plateau_psd.png", protocol
        )
        plt.close(fig)
        payload = {"freq": np.asarray(freq_ref, dtype=float)}
        for key, curve in r_curves.items():
            payload[f"r_{key}"] = np.asarray(curve["median"], dtype=float)
            if curve.get("q1") is not None:
                payload[f"r_{key}_q1"] = np.asarray(curve["q1"], dtype=float)
                payload[f"r_{key}_q3"] = np.asarray(curve["q3"], dtype=float)
        for key, spec in psd_panels.items():
            payload[f"pre_{key}"] = np.asarray(spec["prestim"], dtype=float)
            payload[f"plat_{key}"] = np.asarray(spec["plateau"], dtype=float)
        np.savez(output_root / "spectra.npz", **payload)

    (output_root / "run.json").write_text(
        json.dumps(
            {
                "n_paired_runs": len(pairs),
                "n_trial_rows": len(trial_rows),
                "n_participants": sorted({row["bids_id"] for row in participant_rows}),
                "methods": method_rows,
                "posterior": list(protocol.posterior_channels),
                "prestimulus_seconds": list(protocol.prestimulus_seconds),
                "plateau_seconds": list(protocol.plateau_seconds),
                "band_hz": list(protocol.band_hz),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {output_root}")
    for row in method_rows:
        print(
            f"  {row['method']}: median E={row['median_E']:.3f} dB, "
            f"vs FASTR={row['median_improvement_vs_fastr']:.3f} dB, "
            f"residual={row['median_cardiac_residual']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
