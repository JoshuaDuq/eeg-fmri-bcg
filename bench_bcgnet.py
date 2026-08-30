"""Per-sample timing for the BCGNet GRU. Run on any machine; compare the numbers.

    uv run python bench_bcgnet.py

Reports ms per training sample (fwd+bwd) and per validation sample (fwd only)
at batch 1 and batch 16, plus whether TF sees a GPU and whether the GRU layers
are eligible for the fused cuDNN kernel.
"""
import os, sys, time
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "bcgnet", "vendor"))

import numpy as np
import tensorflow as tf
from models.default_models import RNNModel

N_STEPS, N_IN, N_OUT = 300, 1, 63     # 3 s @ 100 Hz, ECG -> 63 EEG channels
REPS = 12

print(f"tf {tf.__version__}  keras {tf.keras.__version__}")
gpus = tf.config.list_physical_devices("GPU")
print(f"GPUs visible: {[g.name for g in gpus] or 'NONE (CPU only)'}")
print(f"CUDA build: {tf.test.is_built_with_cuda()}")

m = RNNModel(n_input=N_IN, n_output=N_OUT, lr=1e-3)
m.init_model()
m.compile_model()

# cuDNN eligibility, per the 7 conditions in _model_tf_v2's docstring
def _grus(model):
    for layer in model.layers:
        for cand in (getattr(layer, "forward_layer", None),
                     getattr(layer, "backward_layer", None),
                     getattr(layer, "layer", None), layer):
            if cand is not None and cand.__class__.__name__ == "GRU":
                yield cand

for g in _grus(m.model):
    ok = (g.activation.__name__ == "tanh"
          and g.recurrent_activation.__name__ == "sigmoid"
          and g.recurrent_dropout == 0
          and not g.unroll and g.use_bias and g.reset_after)
    print(f"  GRU units={g.units:3d} cudnn_eligible={ok}")
print("  (cudnn_eligible=False on the GPU box means NO fused kernel -> expect little or no speedup)")

def timeit(fn, reps=REPS):
    fn(); fn()                      # warm up / trace
    t = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); t.append(time.perf_counter() - t0)
    return float(np.median(t))

rng = np.random.default_rng(0)
print()
for bs in (1, 16):
    x = rng.standard_normal((bs, N_STEPS, N_IN)).astype("float32")
    y = rng.standard_normal((bs, N_STEPS, N_OUT)).astype("float32")
    train = timeit(lambda: m.model.train_on_batch(x, y))
    fwd = timeit(lambda: m.model.predict_on_batch(x))
    print(f"batch {bs:2d}: train {train*1000:8.2f} ms/step = {train/bs*1000:7.2f} ms/sample"
          f"   |  fwd {fwd*1000:8.2f} ms/step = {fwd/bs*1000:6.2f} ms/sample")

print()
print("Projected epoch for sub-0015 (760 train samples, 163 val samples @ batch 1):")
for bs in (1, 16):
    x = rng.standard_normal((bs, N_STEPS, N_IN)).astype("float32")
    y = rng.standard_normal((bs, N_STEPS, N_OUT)).astype("float32")
    tr = timeit(lambda: m.model.train_on_batch(x, y)) / bs
    x1 = rng.standard_normal((1, N_STEPS, N_IN)).astype("float32")
    fw1 = timeit(lambda: m.model.predict_on_batch(x1))
    print(f"  batch {bs:2d}: train {760*tr:6.1f} s + val {163*fw1:5.1f} s = {760*tr + 163*fw1:6.1f} s/epoch")
