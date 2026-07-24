# Optimizing Edge Automation: The Trade-offs of Quantization and Pruning in Ultra-Lightweight TinyML Topologies

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)

An empirical research study evaluating the impact of **Post-Training INT8 Quantization (PTQ)** and **Constant Magnitude Weight Pruning** on ultra-lightweight neural network topologies (< 500 parameters). Designed for smart irrigation edge nodes, this project demonstrates the **"Optimization Floor"**—a phenomenon where framework metadata overhead outpaces mathematical weight savings in micro-scale networks on host hardware.

---

## 📌 Key Empirical Findings

| Architecture Model | Binary File Size | Footprint Variation | Execution Latency (x86) | Relative Speedup |
| :--- | :--- | :--- | :--- | :--- |
| **Standard Baseline (Float32)** | **2.20 KB** | *Control / Reference* | **859.90 ns** | **1.00x** |
| **INT8 Quantized** | **2.29 KB** | +4.09% (Slight Growth) | **972.30 ns** | **0.88x (13% Slower)** |
| **50% Magnitude Pruned** | **4.74 KB** | +115.45% (Inflated) | *N/A (Sparse Mask)* | *N/A* |

### Key Takeaways
1. **Metadata Overhead Trap:** For networks with under 1,000 parameters, fixed-size TFLite flatbuffer metadata (zero-point offsets, quantization scales, layer indexing) takes up more space than the saved weight bytes.
2. **Dequantization Penalty:** On host CPU architectures equipped with native Floating-Point Units (FPUs) and XNNPACK delegates, converting $Int8$ tensors back to $Float32$ introduces processing overhead that degrades inference speed on micro-topologies.

---

## 📁 Repository Structure

```text
.
├── train_baseline.py      # Generates synthetic sensor dataset & trains baseline MLP
├── optimize_model.py      # Appraisals for INT8 quantization and 50% magnitude pruning
├── benchmark_speed.py     # Measures sub-microsecond latency across generated .tflite models
├── requirements.txt       # Environment dependencies
└── README.md              # Project documentation
