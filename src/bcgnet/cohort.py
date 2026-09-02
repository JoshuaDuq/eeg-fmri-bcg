"""Discover FASTR subjects and train one BCGNet model per subject."""

from __future__ import annotations

import csv
import json
import os
import re
import sys
import time
import traceback
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from bcgstudy.discovery import iter_subjects

from .config import BCGNetConfig, ComputeConfig, ConfigurationError
from .runtime import VENDOR_ROOT, prepare_vendor_imports

_COHORT_SUMMARY_FIELDS = (
    "bids_id",
    "status",
    "n_runs",
    "label",
    "run",
    "stem",
    "n_good",
    "end_epoch",
    "runtime_seconds",
    "rms_raw",
    "rms_bcgnet",
    "delta_bcgnet_ratio",
    "theta_bcgnet_ratio",
    "alpha_bcgnet_ratio",
)
STAGED_NAMING_FORMAT = "{}_{}_raw"


class _Tee:
    def __init__(self, path: Path):
        self.file = path.open("w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)
        self.file.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        self.file.close()


def discover_subjects(config: BCGNetConfig) -> list[dict]:
    if not config.paths.fastr_root.is_dir():
        raise ConfigurationError(
            f"paths.fastr_root is not a directory: {config.paths.fastr_root}"
        )
    subjects = []
    for bids_id, str_sub, recordings in iter_subjects(
        config.paths.fastr_root,
        include=config.subjects.include,
        exclude=config.subjects.exclude,
        run_pattern=config.naming.run_pattern,
    ):
        subjects.append(
            {
                "bids_id": bids_id,
                "str_sub": str_sub,
                "recordings": [
                    {
                        "label": recording.label,
                        "run": recording.run,
                        "stem": recording.stem,
                        "fastr_vhdr": str(recording.path),
                    }
                    for recording in recordings
                ],
            }
        )
    return subjects


def run_count(spec: dict) -> int:
    return sum(1 for item in spec["recordings"] if item["run"] is not None)


def _rewrite_vhdr(src_vhdr: Path, dst_vhdr: Path) -> None:
    text = src_vhdr.read_text(encoding="utf-8", errors="replace")
    dst_eeg = dst_vhdr.with_suffix(".eeg").name
    dst_vmrk = dst_vhdr.with_suffix(".vmrk").name
    text = re.sub(r"(?m)^DataFile=.*$", f"DataFile={dst_eeg}", text)
    text = re.sub(r"(?m)^MarkerFile=.*$", f"MarkerFile={dst_vmrk}", text)
    dst_vhdr.parent.mkdir(parents=True, exist_ok=True)
    dst_vhdr.write_text(text, encoding="utf-8")
    for suffix in (".eeg", ".vmrk"):
        src = src_vhdr.with_suffix(suffix)
        dst = dst_vhdr.with_suffix(suffix)
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(src, dst)


def staged_vhdr(output_root: Path, str_sub: str, label: str) -> Path:
    return (
        output_root
        / "staged"
        / "raw_data"
        / str_sub
        / f"{STAGED_NAMING_FORMAT.format(str_sub, label)}.vhdr"
    )


def stage_subject(spec: dict, output_root: Path) -> None:
    str_sub = spec["str_sub"]
    for recording in spec["recordings"]:
        _rewrite_vhdr(
            Path(recording["fastr_vhdr"]),
            staged_vhdr(output_root, str_sub, recording["label"]),
        )


def _band_metrics(dataset, vendor_cfg, mode: str = "test") -> list[dict]:
    from dataset.dataset_utils import compute_band_power, compute_psd

    raw_set, _eval_set, cleaned_set, _ = dataset._get_set_data(mode=mode)
    f_raw, pxx_raw, _, _ = compute_psd(raw_set)
    f_cln, pxx_cln, _, _ = compute_psd(cleaned_set)
    bands = [
        ("delta", vendor_cfg.cutoff_low_delta, vendor_cfg.cutoff_high_delta),
        ("theta", vendor_cfg.cutoff_low_theta, vendor_cfg.cutoff_high_theta),
        ("alpha", vendor_cfg.cutoff_low_alpha, vendor_cfg.cutoff_high_alpha),
    ]
    rows = []
    for name, lo, hi in bands:
        raw_p = float(compute_band_power(f_raw, pxx_raw, lo, hi))
        cln_p = float(compute_band_power(f_cln, pxx_cln, lo, hi))
        rows.append(
            {
                "band": name,
                "raw": raw_p,
                "bcgnet": cln_p,
                "bcgnet_ratio": cln_p / raw_p if raw_p else None,
            }
        )
    return rows


def worker_environment(config: BCGNetConfig) -> dict[str, str]:
    threads = str(config.compute.threads_per_worker)
    environment = {
        "CUDA_VISIBLE_DEVICES": config.compute.cuda_visible_devices,
        "OMP_NUM_THREADS": threads,
        "TF_NUM_INTRA_OP_THREADS": threads,
        "TF_NUM_INTER_OP_THREADS": "1",
        "TF_ENABLE_ONEDNN_OPTS": "0",
    }
    if config.compute.use_gpu:
        environment["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    return environment


def require_device(compute: ComputeConfig, visible_gpus: Sequence[object]) -> None:
    if compute.use_gpu and not visible_gpus:
        raise ConfigurationError(
            f"compute.device is {compute.device!r} but TensorFlow reports no GPU. "
            "TensorFlow has had no native-Windows GPU build since 2.10; on "
            "Windows run under WSL2 with tensorflow[and-cuda]. Set "
            "compute.device: cpu to train on the CPU instead."
        )
    if compute.use_gpu and not visible_gpus:
        raise ConfigurationError(
            f"compute.device is {compute.device!r} but TensorFlow reports no GPU. "
            "TensorFlow has had no native-Windows GPU build since 2.10; on "
            "Windows run under WSL2 with tensorflow[and-cuda]. Set "
            "compute.device: cpu to train on the CPU instead."
        )


def _initial_result(spec: dict) -> dict:
    return {
        "bids_id": spec["bids_id"],
        "str_sub": spec["str_sub"],
        "status": "error",
        "n_runs": run_count(spec),
        "n_recordings": len(spec["recordings"]),
        "recordings": [
            {
                "label": recording["label"],
                "run": recording["run"],
                "stem": recording["stem"],
            }
            for recording in spec["recordings"]
        ],
    }


def _completed_result(path: Path, *, resume: bool) -> dict | None:
    if not resume or not path.exists():
        return None
    previous = json.loads(path.read_text(encoding="utf-8"))
    if previous.get("status") != "ok":
        return None
    previous["skipped"] = True
    return previous


def _configure_vendor(get_config, config: BCGNetConfig, str_sub: str):
    output_root = config.paths.output_root
    vendor_config = get_config(filename=VENDOR_ROOT / "config" / "default_config.yaml")
    vendor_config.d_root = VENDOR_ROOT
    vendor_config.d_data = output_root / "staged" / "raw_data"
    vendor_config.d_model = output_root / "trained_model" / str_sub
    vendor_config.d_output = output_root / "cleaned_data"
    vendor_config.d_eval = None
    vendor_config.str_eval = None
    vendor_config.num_epochs = config.training.num_epochs
    vendor_config.es_patience = config.training.es_patience
    vendor_config.batch_size = config.training.batch_size
    vendor_config.lr = config.training.learning_rate
    vendor_config.new_fs = config.preprocess.new_fs
    vendor_config.len_epoch = round(config.preprocess.len_epoch)
    vendor_config.mad_threshold = config.preprocess.mad_threshold
    vendor_config.per_training = config.preprocess.per_training
    vendor_config.per_valid = config.preprocess.per_valid
    vendor_config.per_test = config.preprocess.per_test
    vendor_config.str_ecg_channel = config.preprocess.ecg_channel
    vendor_config.input_file_naming_format = STAGED_NAMING_FORMAT
    return vendor_config


def _collect_metrics(session, recordings: list[dict], vendor_config) -> list[dict]:
    metrics = []
    for dataset, recording in zip(session.vec_dataset, recordings, strict=True):
        rms = dataset.rms_results.get("test")
        row = {
            "label": recording["label"],
            "run": recording["run"],
            "stem": recording["stem"],
            "n_good": len(dataset.vec_idx_good_epochs),
        }
        if rms is not None:
            row["rms_raw"] = float(rms[0])
            row["rms_bcgnet"] = float(rms[2])
        try:
            row["bands"] = _band_metrics(dataset, vendor_config, mode="test")
        except Exception as error:
            row["bands_error"] = f"{type(error).__name__}: {error}"
        metrics.append(row)
    return metrics


def _write_cleaned_recordings(session, spec: dict, config: BCGNetConfig) -> None:
    from .export import bcgnet_output_vhdr, write_bcgnet_recording

    for dataset, recording in zip(session.vec_dataset, spec["recordings"], strict=True):
        cleaned = (
            dataset.orig_cleaned_dataset
            if dataset.resampled
            else dataset.cleaned_dataset
        )
        source = Path(recording["fastr_vhdr"])
        write_bcgnet_recording(
            cleaned,
            source,
            bcgnet_output_vhdr(config.paths.output_root, spec["bids_id"], source),
            overwrite=config.training.overwrite,
        )


def _success_result(session, metrics: list[dict], runtime_seconds: float) -> dict:
    return {
        "status": "ok",
        "end_epoch": (
            int(session.end_epoch) if session.end_epoch is not None else None
        ),
        "loss_last": (
            float(session.m.history["loss"][-1]) if session.m is not None else None
        ),
        "val_loss_last": (
            float(session.m.history["val_loss"][-1]) if session.m is not None else None
        ),
        "metrics": metrics,
        "runtime_seconds": runtime_seconds,
    }


def process_subject(spec: dict, config: BCGNetConfig) -> dict:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.update(worker_environment(config))

    prepare_vendor_imports()

    import matplotlib

    matplotlib.use("Agg")
    from config import get_config
    from session import Session

    bids_id = spec["bids_id"]
    str_sub = spec["str_sub"]
    t0 = time.time()
    result = _initial_result(spec)
    output_root = config.paths.output_root
    results_dir = output_root / "results"
    log_dir = output_root / "logs"
    results_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    out_json = results_dir / f"{bids_id}.json"
    log_path = log_dir / f"{bids_id}.log"

    previous = _completed_result(out_json, resume=config.training.resume)
    if previous is not None:
        return previous

    tee = _Tee(log_path)
    old_stdout = sys.stdout
    sys.stdout = tee
    try:
        stage_subject(spec, output_root)
        vendor_cfg = _configure_vendor(get_config, config, str_sub)

        import tensorflow as tf

        require_device(config.compute, tf.config.list_physical_devices("GPU"))

        try:
            tf.config.threading.set_intra_op_parallelism_threads(
                config.compute.threads_per_worker
            )
            tf.config.threading.set_inter_op_parallelism_threads(1)
        except RuntimeError:
            pass

        vec_idx_run = [recording["label"] for recording in spec["recordings"]]
        print(
            f"=== {bids_id} recordings={vec_idx_run} "
            f"runs={run_count(spec)} tf={tf.__version__} "
            f"device={config.compute.device} "
            f"threads={config.compute.threads_per_worker} "
            f"epochs={config.training.num_epochs} batch={config.training.batch_size}"
        )
        session = Session(
            str_sub=str_sub,
            vec_idx_run=vec_idx_run,
            str_arch=config.training.architecture,
            random_seed=config.training.random_seed,
            verbose=2,
            overwrite=config.training.overwrite,
            cv_mode=False,
            cfg=vendor_cfg,
        )
        session.load_all_dataset()
        session.prepare_training()
        session.train()
        session.clean()
        session.evaluate(mode="test")

        metrics = _collect_metrics(session, spec["recordings"], vendor_cfg)

        if config.training.save_model:
            session.save_model()
        if config.training.save_data:
            _write_cleaned_recordings(session, spec, config)
        if config.training.save_figures:
            fig_dir = output_root / "figures" / str_sub
            fig_dir.mkdir(parents=True, exist_ok=True)
            session.plot_training_history(p_figure=fig_dir)

        result.update(_success_result(session, metrics, time.time() - t0))
        print(f"=== {bids_id} DONE in {result['runtime_seconds']:.1f}s")
    except Exception:
        result["traceback"] = traceback.format_exc()
        result["runtime_seconds"] = time.time() - t0
        print(f"=== {bids_id} FAILED")
        print(result["traceback"])
    finally:
        sys.stdout = old_stdout
        tee.close()
        out_json.write_text(json.dumps(result, indent=2))
    return result


def write_cohort_summary(results: list[dict], output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    summary_json = output_root / "cohort_summary.json"
    summary_csv = output_root / "cohort_summary.csv"
    rows = []
    for result in results:
        if result.get("status") != "ok":
            rows.append(
                {
                    "bids_id": result.get("bids_id"),
                    "status": result.get("status"),
                    "n_runs": result.get("n_runs"),
                    "runtime_seconds": result.get("runtime_seconds"),
                }
            )
            continue
        for metric in result.get("metrics") or []:
            bands = {band["band"]: band for band in metric.get("bands") or []}
            rows.append(
                {
                    "bids_id": result["bids_id"],
                    "status": result["status"],
                    "label": metric.get("label"),
                    "run": metric.get("run"),
                    "stem": metric.get("stem"),
                    "n_good": metric.get("n_good"),
                    "end_epoch": result.get("end_epoch"),
                    "runtime_seconds": result.get("runtime_seconds"),
                    "rms_raw": metric.get("rms_raw"),
                    "rms_bcgnet": metric.get("rms_bcgnet"),
                    "delta_bcgnet_ratio": (bands.get("delta") or {}).get(
                        "bcgnet_ratio"
                    ),
                    "theta_bcgnet_ratio": (bands.get("theta") or {}).get(
                        "bcgnet_ratio"
                    ),
                    "alpha_bcgnet_ratio": (bands.get("alpha") or {}).get(
                        "bcgnet_ratio"
                    ),
                }
            )
    summary_json.write_text(
        json.dumps({"n_subjects": len(results), "results": results}, indent=2)
    )
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_COHORT_SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return summary_csv


def _worker_from_path(spec: dict, config_path: str) -> dict:
    from .config import load_config

    return process_subject(spec, load_config(config_path))


def run_cohort(config: BCGNetConfig, config_path: Path) -> list[dict]:
    subjects = discover_subjects(config)
    print(
        f"cohort: {len(subjects)} subjects, workers={config.compute.workers}, "
        f"device={config.compute.device}, "
        f"threads/worker={config.compute.threads_per_worker}"
    )
    for spec in subjects:
        print(
            f"  {spec['bids_id']}: {run_count(spec)} runs, "
            f"{len(spec['recordings'])} recordings"
        )
    results: list[dict] = []
    with ProcessPoolExecutor(max_workers=config.compute.workers) as pool:
        futures = {
            pool.submit(_worker_from_path, spec, str(config_path)): spec["bids_id"]
            for spec in subjects
        }
        for future in as_completed(futures):
            bids_id = futures[future]
            try:
                result = future.result()
            except Exception:
                result = {
                    "bids_id": bids_id,
                    "status": "error",
                    "traceback": traceback.format_exc(),
                }
            results.append(result)
            n_ok = sum(1 for item in results if item.get("status") == "ok")
            n_err = sum(1 for item in results if item.get("status") != "ok")
            print(
                f"progress {len(results)}/{len(subjects)} "
                f"ok={n_ok} err={n_err} last={bids_id}:{result.get('status')}"
            )
            write_cohort_summary(results, config.paths.output_root)
    write_cohort_summary(results, config.paths.output_root)
    return results
