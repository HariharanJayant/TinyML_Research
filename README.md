# Optimizing Edge Automation: The Trade-offs of Quantization and Pruning in Ultra-Lightweight TinyML Topologies

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)

An empirical study of **post-training INT8 quantization (PTQ)** and **50% constant magnitude weight pruning** on a single 161-parameter neural network (3-12-8-1 MLP) for a smart irrigation edge node, compared with a decision tree and logistic regression. For a network this small, the fixed overhead of the TFLite format outweighs the savings from these optimizations on host hardware.

---

## Results (seed 42, Intel Xeon, 2.20 GHz, TensorFlow 2.21.0, run on Google Colab)

| Model | Parameters | File size | Size change | TFLM arena | Test accuracy | Latency (x86) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Float32 baseline | 161 | 2.66 KB | control | 1.55 KB | 94.00% | 1,896.76 ns |
| INT8 quantized | 161 | 3.06 KB | +15.12% | 1.73 KB | 94.00% | 1,983.22 ns |
| 50% pruned (stripped) | 161 (85 nonzero) | 2.66 KB | +0.00% | 1.55 KB | 93.80% | 1,911.90 ns |
| Decision tree (depth 4) | 29 nodes | 317 B (C arrays) | -88.36% | n/a | 94.60% | 686.41 ns |
| Logistic regression | 4 | 16 B (C arrays) | -99.41% | n/a | 93.80% | 677.91 ns |

Gzipped sizes: baseline 1.48 KB, INT8 1.34 KB, pruned 1.27 KB.
Pruned model exported **without** `strip_pruning`: 7.11 KB (+167.11%).
Accuracy over five training seeds (mean ± SD): baseline 93.52 ± 0.36%, INT8 93.44 ± 0.36%, pruned 92.20 ± 2.53%.
Classical-model latencies are dominated by the Python-to-C call overhead (~682 ns for an empty call).

### Key takeaways
1. **Quantization overhead:** for this 161-parameter network, INT8 quantization made the .tflite file and the TFLM arena larger and host inference slower, because fixed per-tensor metadata (scales, zero points) costs more than the bytes saved on weights.
2. **Pruning needs a sparse format:** after `strip_pruning`, half the weights are zero but TFLite stores them densely, so the file size does not change; the saving only appears after compression.
3. **Classical baselines are strong:** a depth-4 decision tree was the most accurate model and needs a few hundred bytes.

---

## Reproduce

```bash
pip install -r requirements.txt
TF_USE_LEGACY_KERAS=1 python run_experiment.py
```

`run_experiment.py` writes `results.json`, `results_seeds.json`, the `.tflite` models and the generated C code. `tflite-micro` only installs on Linux, so run it on Linux or Google Colab. Latency depends on the machine, so your timings will differ.

## Repository structure

```text
.
├── run_experiment.py      # Full pipeline: noisy dataset, training, INT8 PTQ, pruning, baselines, all measurements
├── requirements.txt       # Environment dependencies
└── README.md              # Project documentation
```

The earlier scripts (`train_baseline.py`, `optimize_model.py`, `benchmark_speed.py`) are replaced by `run_experiment.py`. `optimize_model.py` converted an untrained copy of the network and exported the pruned model without `strip_pruning`, so its results are not valid.
