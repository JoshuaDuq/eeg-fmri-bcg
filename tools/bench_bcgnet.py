"""Measure per-sample training and inference time for the BCGNet GRU.

Run with ``uv run python tools/bench_bcgnet.py`` on each target machine.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterator
from functools import partial

import numpy as np

from bcgnet.runtime import prepare_vendor_imports

_TIME_STEPS = 300
_INPUT_CHANNELS = 1
_OUTPUT_CHANNELS = 63
_REPETITIONS = 12
_BATCH_SIZES = (1, 16)


def _median_runtime(operation: Callable[[], object]) -> float:
    operation()
    operation()
    runtimes = []
    for _ in range(_REPETITIONS):
        start = time.perf_counter()
        operation()
        runtimes.append(time.perf_counter() - start)
    return float(np.median(runtimes))


def _gru_layers(model) -> Iterator[object]:
    for layer in model.layers:
        candidates = (
            getattr(layer, "forward_layer", None),
            getattr(layer, "backward_layer", None),
            getattr(layer, "layer", None),
            layer,
        )
        yield from (
            candidate
            for candidate in candidates
            if candidate is not None and candidate.__class__.__name__ == "GRU"
        )


def _is_cudnn_eligible(layer) -> bool:
    return (
        layer.activation.__name__ == "tanh"
        and layer.recurrent_activation.__name__ == "sigmoid"
        and layer.recurrent_dropout == 0
        and not layer.unroll
        and layer.use_bias
        and layer.reset_after
    )


def _random_batch(
    generator: np.random.Generator, batch_size: int
) -> tuple[np.ndarray, np.ndarray]:
    inputs = generator.standard_normal(
        (batch_size, _TIME_STEPS, _INPUT_CHANNELS)
    ).astype("float32")
    targets = generator.standard_normal(
        (batch_size, _TIME_STEPS, _OUTPUT_CHANNELS)
    ).astype("float32")
    return inputs, targets


def _report_batch_timings(model, generator: np.random.Generator) -> None:
    for batch_size in _BATCH_SIZES:
        inputs, targets = _random_batch(generator, batch_size)
        train = _median_runtime(partial(model.train_on_batch, inputs, targets))
        forward = _median_runtime(partial(model.predict_on_batch, inputs))
        print(
            f"batch {batch_size:2d}: train {train * 1000:8.2f} ms/step "
            f"= {train / batch_size * 1000:7.2f} ms/sample | "
            f"fwd {forward * 1000:8.2f} ms/step "
            f"= {forward / batch_size * 1000:6.2f} ms/sample"
        )


def _report_projected_epoch(model, generator: np.random.Generator) -> None:
    print("Projected epoch for 760 training and 163 validation samples:")
    validation, _targets = _random_batch(generator, 1)
    for batch_size in _BATCH_SIZES:
        inputs, targets = _random_batch(generator, batch_size)
        train_per_sample = (
            _median_runtime(partial(model.train_on_batch, inputs, targets)) / batch_size
        )
        validation_per_sample = _median_runtime(
            partial(model.predict_on_batch, validation)
        )
        training_seconds = 760 * train_per_sample
        validation_seconds = 163 * validation_per_sample
        print(
            f"  batch {batch_size:2d}: train {training_seconds:6.1f} s + "
            f"validation {validation_seconds:5.1f} s = "
            f"{training_seconds + validation_seconds:6.1f} s/epoch"
        )


def main() -> int:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    prepare_vendor_imports()

    import tensorflow as tf
    from models.default_models import RNNModel

    print(f"tf {tf.__version__} keras {tf.keras.__version__}")
    visible_gpus = tf.config.list_physical_devices("GPU")
    print(f"GPUs visible: {[gpu.name for gpu in visible_gpus] or 'NONE (CPU only)'}")
    print(f"CUDA build: {tf.test.is_built_with_cuda()}")

    wrapper = RNNModel(n_input=_INPUT_CHANNELS, n_output=_OUTPUT_CHANNELS, lr=1e-3)
    wrapper.init_model()
    wrapper.compile_model()
    for layer in _gru_layers(wrapper.model):
        print(
            f"  GRU units={layer.units:3d} cudnn_eligible={_is_cudnn_eligible(layer)}"
        )

    generator = np.random.default_rng(0)
    _report_batch_timings(wrapper.model, generator)
    print()
    _report_projected_epoch(wrapper.model, generator)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
