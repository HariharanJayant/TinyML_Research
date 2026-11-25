"""
Full experiment for "Evaluating optimization trade-offs in ultra-lightweight
neural networks for edge automation".

One script, one run: generates the noisy synthetic dataset, trains the
3-12-8-1 MLP, converts the TRAINED model to float32 TFLite, full-INT8 TFLite
(post-training quantization) and 50%-magnitude-pruned TFLite, measures test
accuracy, file size, TFLite Micro arena memory and host latency for each, and
trains decision tree and logistic regression baselines.

Usage:  TF_USE_LEGACY_KERAS=1 python run_experiment.py
Writes: results.json, results_seeds.json and the .tflite / C model files.
"""
import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")  # tfmot needs Keras 2 (tf_keras)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import ctypes, gzip, json, platform, re, statistics, subprocess, sys, time
import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

OUT = os.path.dirname(os.path.abspath(__file__))
DATA_SEED = 42
N_SAMPLES = 5000
# Gaussian sensor noise (standard deviations) added to the readings the model sees.
# Labels are computed from the true values, so noise creates overlapping class boundaries.
NOISE_SD = {"temperature": 1.5, "humidity": 3.0, "soil_moisture": 3.0}
EPOCHS, PATIENCE, BATCH = 100, 15, 32
PRUNE_EPOCHS = 10
LAT_ITERS, LAT_REPEATS = 10000, 7


# ---------------------------------------------------------------- data
def make_data(seed=DATA_SEED, n=N_SAMPLES):
    rng = np.random.default_rng(seed)
    t = rng.uniform(15, 45, n)
    h = rng.uniform(20, 90, n)
    sm = rng.uniform(10, 80, n)
    y = ((sm < 30) | ((sm < 45) & (t > 35))).astype(np.float32)  # valve rule on TRUE values
    X = np.c_[t + rng.normal(0, NOISE_SD["temperature"], n),
              h + rng.normal(0, NOISE_SD["humidity"], n),
              sm + rng.normal(0, NOISE_SD["soil_moisture"], n)].astype(np.float32)
    # 80 / 10 / 10 stratified split
    Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    Xva, Xte, yva, yte = train_test_split(Xtmp, ytmp, test_size=0.5, random_state=seed, stratify=ytmp)
    return Xtr, Xva, Xte, ytr, yva, yte


def build_mlp():
    m = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(3,)),
        tf.keras.layers.Dense(12, activation="relu"),
        tf.keras.layers.Dense(8, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    m.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return m


# ---------------------------------------------------------------- TFLite helpers
def to_float_tflite(model):
    return tf.lite.TFLiteConverter.from_keras_model(model).convert()


def to_int8_tflite(model, calib):
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    def rep():
        for i in range(len(calib)):
            yield [calib[i:i + 1]]
    conv.representative_dataset = rep
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    return conv.convert()


def make_interp(blob):
    it = tf.lite.Interpreter(model_content=blob)
    it.allocate_tensors()
    return it


def quant_in(it, x):
    d = it.get_input_details()[0]
    if d["dtype"] == np.int8:
        sc, zp = d["quantization"]
        return np.clip(np.round(x / sc + zp), -128, 127).astype(np.int8)
    return x.astype(np.float32)


def tflite_accuracy(blob, X, y):
    it = make_interp(blob)
    i, o = it.get_input_details()[0], it.get_output_details()[0]
    correct = 0
    for x, t in zip(X, y):
        it.set_tensor(i["index"], quant_in(it, x[None]))
        it.invoke()
        out = it.get_tensor(o["index"]).astype(np.float32)
        if o["dtype"] == np.int8:
            sc, zp = o["quantization"]
            out = (out - zp) * sc
        correct += int((out.ravel()[0] > 0.5) == bool(t))
    return correct / len(y)


def tflite_latency_ns(blob, x):
    """Median over LAT_REPEATS of the mean time of set_tensor + invoke (same as benchmark_speed.py)."""
    it = make_interp(blob)
    i = it.get_input_details()[0]
    xin = quant_in(it, x[None])
    for _ in range(100):  # warm-up
        it.set_tensor(i["index"], xin); it.invoke()
    reps = []
    for _ in range(LAT_REPEATS):
        t0 = time.perf_counter_ns()
        for _ in range(LAT_ITERS):
            it.set_tensor(i["index"], xin); it.invoke()
        reps.append((time.perf_counter_ns() - t0) / LAT_ITERS)
    return statistics.median(reps)


def tflm_arena_bytes(path):
    """Total TensorFlow Lite Micro arena use, read from the RecordingMicroAllocator report."""
    code = ("from tflite_micro.python.tflite_micro import runtime;"
            f"runtime.Interpreter.from_file({path!r}).print_allocations()")
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    out = p.stdout + p.stderr
    return int(re.search(r"Arena allocation total (\d+) bytes", out).group(1))


def gz_size(blob):
    return len(gzip.compress(blob, compresslevel=9))


# ---------------------------------------------------------------- pruning
def prune(trained, Xtr, ytr, Xva, yva):
    # Wrap the TRAINED model (weights are kept), then fine-tune while the 50% mask is applied
    pm = tfmot.sparsity.keras.prune_low_magnitude(
        trained, pruning_schedule=tfmot.sparsity.keras.ConstantSparsity(0.5, begin_step=0, frequency=100))
    pm.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    pm.fit(Xtr, ytr, epochs=PRUNE_EPOCHS, batch_size=BATCH, validation_data=(Xva, yva),
           callbacks=[tfmot.sparsity.keras.UpdatePruningStep()], verbose=0)
    return pm


def sparsity_report(model):
    rep = {}
    for layer in model.layers:
        if hasattr(layer, "kernel"):
            k = layer.kernel.numpy()
            rep[layer.name] = float((k == 0).mean())
    return rep


# ---------------------------------------------------------------- classical baselines in C
C_HARNESS = r"""
#include <stdint.h>
#include "%(header)s"
int predict(const float *x) { return %(call)s; }
"""


def build_c_model(name, c_source_or_header, call, is_header=True):
    d = os.path.join(OUT, "c_models"); os.makedirs(d, exist_ok=True)
    hdr = os.path.join(d, f"{name}.h")
    with open(hdr, "w") as f:
        f.write(c_source_or_header)
    src = os.path.join(d, f"{name}_lib.c")
    with open(src, "w") as f:
        f.write(C_HARNESS % {"header": f"{name}.h", "call": call})
    so = os.path.join(d, f"lib{name}.so")
    subprocess.run(["gcc", "-O2", "-shared", "-fPIC", "-I", d, src, "-o", so], check=True)
    obj = os.path.join(d, f"{name}.o")
    subprocess.run(["gcc", "-Os", "-c", "-I", d, src, "-o", obj], check=True)
    # bytes of model data (read-only + initialised data sections) in the compiled object
    out = subprocess.run(["size", "-A", obj], capture_output=True, text=True).stdout
    data_bytes = sum(int(m.group(2)) for m in re.finditer(r"^(\.rodata\S*|\.data\S*)\s+(\d+)", out, re.M))
    lib = ctypes.CDLL(so)
    lib.predict.argtypes = [ctypes.POINTER(ctypes.c_float)]
    lib.predict.restype = ctypes.c_int
    return lib, data_bytes


def tree_to_c(dt):
    """Export a fitted sklearn decision tree to C arrays with the exact float thresholds."""
    t = dt.tree_
    n = t.node_count
    feat = [int(f) if f >= 0 else -1 for f in t.feature]
    thr = [float(v) if f >= 0 else 0.0 for v, f in zip(t.threshold, t.feature)]
    cls = [int(np.argmax(v)) for v in t.value[:, 0, :]]
    leaf_class = [int(dt.classes_[c]) for c in cls]
    arr = lambda typ, name, vals, fmt: f"static const {typ} {name}[{n}] = {{" + ", ".join(fmt % v for v in vals) + "};\n"
    return (arr("int8_t", "dt_feat", feat, "%d") + arr("float", "dt_thr", thr, "%.9ef")
            + arr("int16_t", "dt_left", t.children_left, "%d") + arr("int16_t", "dt_right", t.children_right, "%d")
            + arr("uint8_t", "dt_class", leaf_class, "%d")
            + "static inline int dtree_predict(const float *x) {\n  int i = 0;\n"
              "  while (dt_feat[i] >= 0) i = (x[dt_feat[i]] <= dt_thr[i]) ? dt_left[i] : dt_right[i];\n"
              "  return dt_class[i];\n}\n")


def c_accuracy(lib, X, y):
    preds = [lib.predict((ctypes.c_float * 3)(*x)) for x in X]
    return float(np.mean(np.array(preds) == y))


def c_latency_ns(lib, x):
    buf = (ctypes.c_float * 3)(*x)
    for _ in range(100):
        lib.predict(buf)
    reps = []
    for _ in range(LAT_REPEATS):
        t0 = time.perf_counter_ns()
        for _ in range(LAT_ITERS):
            lib.predict(buf)
        reps.append((time.perf_counter_ns() - t0) / LAT_ITERS)
    return statistics.median(reps)


# ---------------------------------------------------------------- one full run
def run_nn(seed, Xtr, Xva, Xte, ytr, yva, yte, save=False):
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    model = build_mlp()
    hist = model.fit(Xtr, ytr, epochs=EPOCHS, batch_size=BATCH, validation_data=(Xva, yva), verbose=0,
                     callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE,
                                                                 restore_best_weights=True)])
    keras_test_acc = float(model.evaluate(Xte, yte, verbose=0)[1])
    calib = Xtr[:100]  # 100 training rows, features only (no labels)
    base = to_float_tflite(model)
    quant = to_int8_tflite(model, calib)
    pm = prune(model, Xtr, ytr, Xva, yva)
    pruned_unstripped = to_float_tflite(pm)          # what the original optimize_model.py exported
    stripped = tfmot.sparsity.keras.strip_pruning(pm)
    pruned = to_float_tflite(stripped)
    r = {
        "seed": seed,
        "epochs_trained": len(hist.history["loss"]),
        "params": int(model.count_params()),
        "keras_test_acc": keras_test_acc,
        "acc": {"baseline": tflite_accuracy(base, Xte, yte),
                "int8": tflite_accuracy(quant, Xte, yte),
                "pruned": tflite_accuracy(pruned, Xte, yte)},
        "sparsity": sparsity_report(stripped),
        "nonzero_params": int(sum(np.count_nonzero(w) for w in stripped.get_weights())),
    }
    if save:
        blobs = {"baseline": base, "int8": quant, "pruned": pruned, "pruned_unstripped": pruned_unstripped}
        for k, b in blobs.items():
            with open(os.path.join(OUT, f"{k}.tflite"), "wb") as f:
                f.write(b)
        r["size_bytes"] = {k: len(b) for k, b in blobs.items()}
        r["gzip_bytes"] = {k: gz_size(b) for k, b in blobs.items()}
        r["arena_bytes"] = {k: tflm_arena_bytes(os.path.join(OUT, f"{k}.tflite")) for k in ("baseline", "int8", "pruned")}
        x = Xte[0]
        r["latency_ns"] = {k: tflite_latency_ns(blobs[k], x) for k in ("baseline", "int8", "pruned")}
        r["acc"]["pruned_unstripped"] = tflite_accuracy(pruned_unstripped, Xte, yte)
    return r


def main():
    Xtr, Xva, Xte, ytr, yva, yte = make_data()
    res = {
        "host": {"cpu": _cpu_name(), "python": platform.python_version(), "tensorflow": tf.__version__,
                 "tfmot": tfmot.__version__ if hasattr(tfmot, "__version__") else "0.8.x"},
        "data": {"n": N_SAMPLES, "noise_sd": NOISE_SD, "split": [len(ytr), len(yva), len(yte)],
                 "frac_open": float(np.r_[ytr, yva, yte].mean())},
    }
    res["nn"] = run_nn(42, Xtr, Xva, Xte, ytr, yva, yte, save=True)

    np.seterr(all="warn")  # a dependency switches NumPy to raise on underflow; sklearn needs the default
    # Decision tree: depth chosen on the validation set
    best = max(range(1, 13), key=lambda d: (DecisionTreeClassifier(max_depth=d, random_state=42)
                                            .fit(Xtr, ytr).score(Xva, yva), -d))
    dt = DecisionTreeClassifier(max_depth=best, random_state=42).fit(Xtr, ytr)
    dt_lib, dt_bytes = build_c_model("dtree", tree_to_c(dt), "dtree_predict(x)")
    # Logistic regression (regularisation C chosen on the validation set)
    bestC = max([0.01, 0.1, 1, 10, 100], key=lambda c: LogisticRegression(C=c, max_iter=5000).fit(Xtr, ytr).score(Xva, yva))
    lr = LogisticRegression(C=bestC, max_iter=5000).fit(Xtr, ytr)
    w, b = lr.coef_[0], lr.intercept_[0]
    lr_c = ("static const float lr_w[3] = {%.9ef, %.9ef, %.9ef};\nstatic const float lr_b = %.9ef;\n"
            "static inline int lr_predict(const float *x) {\n"
            "  float z = lr_b + lr_w[0]*x[0] + lr_w[1]*x[1] + lr_w[2]*x[2];\n  return z > 0.0f;\n}\n") % (*w, b)
    lr_lib, lr_bytes = build_c_model("logreg", lr_c, "lr_predict(x)")
    noop_c = "static inline int noop(const float *x) { return 0; }\n"
    noop_lib, _ = build_c_model("noop", noop_c, "noop(x)")
    x = Xte[0]
    res["classical"] = {
        "decision_tree": {"max_depth": best, "nodes": int(dt.tree_.node_count), "leaves": int(dt.get_n_leaves()),
                          "test_acc_sklearn": float(dt.score(Xte, yte)), "test_acc_c": c_accuracy(dt_lib, Xte, yte),
                          "model_data_bytes": dt_bytes, "latency_ns": c_latency_ns(dt_lib, x)},
        "logistic_regression": {"C": bestC, "params": 4, "test_acc_sklearn": float(lr.score(Xte, yte)),
                                "test_acc_c": c_accuracy(lr_lib, Xte, yte), "model_data_bytes": lr_bytes,
                                "latency_ns": c_latency_ns(lr_lib, x)},
        "ctypes_call_overhead_ns": c_latency_ns(noop_lib, x),
    }
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))

    # Robustness: retrain the network with 5 initialisation seeds (same data)
    seeds = [run_nn(s, Xtr, Xva, Xte, ytr, yva, yte) for s in (42, 43, 44, 45, 46)]
    with open(os.path.join(OUT, "results_seeds.json"), "w") as f:
        json.dump(seeds, f, indent=2)
    for k in ("baseline", "int8", "pruned"):
        a = [s["acc"][k] for s in seeds]
        print(f"{k}: mean {np.mean(a)*100:.2f}%  sd {np.std(a, ddof=1)*100:.2f}  min {min(a)*100:.2f}  max {max(a)*100:.2f}")


def _cpu_name():
    try:
        return re.search(r"model name\s*:\s*(.*)", open("/proc/cpuinfo").read()).group(1)
    except Exception:
        return platform.processor()


if __name__ == "__main__":
    main()
