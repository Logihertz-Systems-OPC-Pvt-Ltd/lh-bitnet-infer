# Changelog — Logihertz BitNet Inference Engine

All notable changes to `lh-bitnet-infer` will be documented in this file.

The format is based on [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-11-13

### Added
- **1.58-Bit Ternary Weight Quantization Engine**:
  - `Absmean` scaling factor calculation: $\gamma = \frac{1}{MK} \sum |W_{i,j}| + 10^{-8}$.
  - Deterministic ternary round-and-clamp mapping to $\widetilde{W}_{i,j} \in \{-1, 0, +1\}$.
  - 2-bit binary byte packing ($c = w + 1$, 4 weights per `uint8_t`), achieving $16\times$ physical memory footprint reduction.
  - Lossless byte unpacking with optional bounds checking.
  - Zero-weight sparsity ratio ($\rho_0$) and Quantization Mean Squared Error (MSE) telemetry.
  - Cryptographic FIPS 180-4 SHA-256 weight integrity checksum generator and validator.
- **Per-Token Activation Quantization & Normalization**:
  - `RMSNorm` pre-activation normalization with learned gain vector $g$ and configurable $\epsilon = 10^{-5}$.
  - `Absmax` 8-bit dynamic activation quantization mapping peak token activation $\eta$ to signed `int8_t` $[-127, +127]$.
- **Multiplication-Free SIMD `BitLinear` Kernel**:
  - Integer ternary addition/subtraction accumulation ($A_i = \sum_{w=+1} \tilde{x}_j - \sum_{w=-1} \tilde{x}_j$).
  - Per-row output floating-point rescaling ($y_i = A_i \cdot \frac{\gamma \cdot \eta}{127.0}$).
  - Hardware SIMD vectorization: AVX2 (`_mm256_sign_epi8`, `_mm256_maddubs_epi16`, `_mm256_add_epi32`) on x86_64 and ARM NEON (`vmull_s8`, `vpadalq_s16`) on ARM64.
  - Portable bit-exact scalar reference path verified for 100% mathematical parity.
- **Secure KV-Cache & Memory Guard**:
  - Bounded KV-cache manager with context length limits and OOM defense.
  - `secure_zero_memory()` wiping sensitive residual activation telemetry on session reset.
- **10-Rule Deterministic Edge LLM Posture Auditor**:
  - Rule registry `LH-BITNET-001` through `LH-BITNET-010` mapping to MITRE CWE definitions.
  - Weighted deduction scoring: $S_{\text{posture}} = \max(0, 100 - \sum \text{weight})$.
  - Exporters: ANSI terminal box table with score banner, Canonical JSON, and OASIS SARIF v2.1.0.
- **Command-Line Interface (`bin/lh-bitnet`)**:
  - `scan <config.json>`: Evaluate edge LLM deployment configuration against 10 posture rules.
  - `quantize <weights.json>`: Quantize FP32 matrix to ternary, pack to 2-bit `.bnet`, and report metrics.
  - `infer <layer.json>`: Execute full forward pass (RMSNorm + Absmax INT8 + BitLinear) and report output.
  - `benchmark`: Measure SIMD vs Scalar GOPS throughput and verify zero difference.
  - `--version` and `--help`.
