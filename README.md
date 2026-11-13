# Logihertz BitNet 1.58-Bit Inference Engine (`lh-bitnet-infer`)

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Language](https://img.shields.io/badge/Language-C%2B%2B17-blue.svg)](Makefile)
[![Standard](https://img.shields.io/badge/SIMD-AVX2%20%7C%20ARM_NEON-orange.svg)](include/bitlinear_kernel.hpp)
[![SARIF](https://img.shields.io/badge/SARIF-OASIS%20v2.1.0-brightgreen.svg)](https://docs.oasis-open.org/sarif/sarif/v2.1.0/sarif-v2.1.0.html)
[![Zero-Dependencies](https://img.shields.io/badge/Dependencies-0%20External-success.svg)](Makefile)

**`lh-bitnet-infer`** (`bin/lh-bitnet`) is an enterprise-grade, clean-room, zero-dependency C++17 1.58-bit ternary neural network (`BitNet b1.58`) inference runtime engineered for sovereign edge AI appliances, robotics, and embedded systems (such as NVIDIA Jetson Orin and industrial ARM64/x86_64 nodes). It provides a multiplication-free SIMD `BitLinear` matrix multiplication kernel, Absmean ternary weight quantization with $16\times$ 2-bit byte packing, Absmax INT8 per-token activation quantization, pre-activation RMSNorm, bounded KV-cache memory isolation, and a 10-rule deterministic Edge LLM posture auditor.

---

## Key Capabilities

1. **100% Zero External Third-Party Library Dependencies**:
   Engineered strictly in ISO C++17 standard library headers (`<vector>`, `<cmath>`, `<string>`, `<fstream>`, `<chrono>`, `<iostream>`). Zero third-party runtime or linear algebra dependencies.
2. **1.58-Bit Ternary Weight Quantization (`Absmean`)**:
   - Computes dynamic per-tensor scaling factor $\gamma = \frac{1}{M \cdot K}\sum |W_{i,j}|$.
   - Quantizes continuous FP32 weights into ternary values $\widetilde{W}_{i,j} \in \{-1, 0, +1\}$ ($\log_2(3) \approx 1.585\text{ bits}$).
   - Dense 2-bit byte packing ($4\text{ weights per byte}$), reducing physical memory footprint by $16\times$ compared to standard FP32.
   - Computes zero-weight sparsity ratio ($\rho_0$) and Quantization Mean Squared Error (MSE).
3. **Multiplication-Free SIMD `BitLinear` Kernel**:
   - Replaces costly floating-point matrix multiplications with signed 32-bit integer addition and subtraction:
     $$A_i = \sum_{j: \widetilde{W}_{i,j} = +1} \tilde{x}_j - \sum_{j: \widetilde{W}_{i,j} = -1} \tilde{x}_j$$
   - Rescales once per output row to FP32: $y_i = A_i \cdot \frac{\gamma \cdot \eta}{127.0}$.
   - Hardware SIMD vectorization: AVX2 (`_mm256_sign_epi8`, `_mm256_maddubs_epi16`, `_mm256_add_epi32`) on x86_64, ARM NEON on ARM64, and bit-exact scalar fallback.
4. **Per-Token Activation Quantization & RMSNorm**:
   - Pre-activation `RMSNorm` with learned gain vector $g$ and $\epsilon = 10^{-5}$.
   - Symmetric signed 8-bit (`int8_t` $[-127, +127]$) activation quantization based on maximum token activation magnitude $\eta$.
5. **Hardware Security & Memory Isolation**:
   - FIPS 180-4 SHA-256 weight integrity verification preventing trojaned model injection.
   - Bounded KV-cache manager with capacity limits and `secure_zero_memory()` wipe on session termination.
6. **10-Rule Deterministic Edge LLM Posture Auditor**:
   - Evaluates deployment manifests against 10 normative rules with weighted deduction scoring.
   - Multi-format exporters: ANSI box table with score banner, Canonical JSON, and OASIS SARIF v2.1.0.

---

## Mathematical Foundations & BitNet b1.58 Architecture

### 1. 1.58-Bit Ternary Weight Quantization (`Absmean`)

For an FP32 weight matrix $W \in \mathbb{R}^{M \times K}$:
$$\gamma = \frac{1}{M \cdot K} \sum_{i=0}^{M-1} \sum_{j=0}^{K-1} |W_{i,j}| + 10^{-8}$$

$$\widetilde{W}_{i,j} = \text{clamp}\left(\text{round}\left(\frac{W_{i,j}}{\gamma}\right), \; -1, \; +1\right) \in \{-1, 0, +1\}$$

### 2. 2-Bit Byte Packing Scheme

Ternary weights $w \in \{-1, 0, +1\}$ are encoded as unsigned 2-bit codes $c = w + 1 \in \{0, 1, 2\}$:
- `-1` $\to$ `00_2` (0)
- ` 0` $\to$ `01_2` (1)
- `+1` $\to$ `10_2` (2)

Four consecutive ternary weights are packed into one `uint8_t` byte:
$$\text{byte}_b = c_{4b} \mid (c_{4b+1} \ll 2) \mid (c_{4b+2} \ll 4) \mid (c_{4b+3} \ll 6)$$

A matrix of size $M \times K$ occupies $\lceil (M \cdot K) / 4 \rceil$ bytes, delivering a $16\times$ compression ratio over FP32.

### 3. Pre-Activation RMSNorm

$$\text{RMSNorm}(x)_j = \frac{x_j}{\sqrt{\frac{1}{K}\sum_{m=0}^{K-1} x_m^2 + \epsilon}} \cdot g_j$$
where $\epsilon = 10^{-5}$ and $g \in \mathbb{R}^K$ is the learned gain vector.

### 4. 8-Bit Per-Token Activation Quantization (`Absmax`)

Given normalized token activation vector $x \in \mathbb{R}^K$:
$$\eta = \max_{0 \le j < K} |x_j| + 10^{-8}$$

$$\tilde{x}_j = \text{clamp}\left(\text{round}\left(x_j \cdot \frac{127.0}{\eta}\right), \; -127, \; +127\right)$$

### 5. Multiplication-Free Forward Pass (`BitLinear`)

The inner product is computed in signed 32-bit integer accumulators:
$$A_i = \sum_{j=0}^{K-1} \widetilde{W}_{i,j} \cdot \tilde{x}_j = \sum_{j: \widetilde{W}_{i,j} = +1} \tilde{x}_j - \sum_{j: \widetilde{W}_{i,j} = -1} \tilde{x}_j$$

Output activations are rescaled once per row $i$:
$$y_i = A_i \cdot \left(\frac{\gamma \cdot \eta}{127.0}\right)$$

---

## Normative 10-Rule Edge LLM Posture Registry

| Rule ID | Severity | CWE | Weight | Scope & Normative Audit Check |
|:---|:---:|:---:|:---:|:---|
| `LH-BITNET-001` | **CRITICAL** | CWE-353 | 25 | Unsigned or unverified model weights (`verify_weight_sha256: false` or `model_signature_verified: false`) |
| `LH-BITNET-002` | **CRITICAL** | CWE-770 | 25 | Unbounded KV-cache memory or missing OOM guard (`max_context_tokens > 8192`, `kv_cache_max_mb > 1024`, or `oom_guard_enabled: false`) |
| `LH-BITNET-003` | **CRITICAL** | CWE-319 | 25 | Unauthenticated or plaintext inference RPC server bound to external interface (`bind_address != "127.0.0.1"` and insecure) |
| `LH-BITNET-004` | **HIGH** | CWE-682 | 10 | Excessive quantization MSE or invalid scale (`quantization_mse_threshold > 0.05` or `absmean_scale_gamma <= 0.0`) |
| `LH-BITNET-005` | **HIGH** | CWE-190 | 10 | Missing INT32 accumulator overflow guard or disabled INT8 activation clipping |
| `LH-BITNET-006` | **HIGH** | CWE-119 | 10 | Unaligned SIMD tensor buffers (`simd_alignment_bytes < 32`) or unchecked 2-bit unpacking |
| `LH-BITNET-007` | **HIGH** | CWE-74 | 10 | Missing prompt token sanitization or unbounded generation length (`max_new_tokens > 2048` or missing stop sequence) |
| `LH-BITNET-008` | **MEDIUM** | CWE-682 | 4 | Invalid RMSNorm epsilon or disabled pre-activation normalization (`pre_rmsnorm_enabled: false` or `rmsnorm_eps <= 0.0`) |
| `LH-BITNET-009` | **MEDIUM** | CWE-226 | 4 | Residual prompt/KV-cache memory not zeroed on session reset or unlocked pages (`zero_kv_cache_on_reset: false`) |
| `LH-BITNET-010` | **LOW** | CWE-755 | 1 | Missing edge SoC thermal throttling guard or non-deterministic sampling seed |

### Posture Scoring Equation

$$S_{\text{posture}} = \max\left(0, 100 - (25\cdot c_{\text{CRIT}} + 10\cdot c_{\text{HIGH}} + 4\cdot c_{\text{MED}} + 1\cdot c_{\text{LOW}})\right)$$

---

## CLI Reference (`bin/lh-bitnet`)

```bash
# 1. Audit edge LLM deployment configuration against 10 posture rules
bin/lh-bitnet scan fixtures/hardened_bitnet_config.json
bin/lh-bitnet scan fixtures/vulnerable_bitnet_config.json --format sarif --output audit.sarif

# 2. Quantize FP32 matrix to 1.58-bit ternary and save packed .bnet binary
bin/lh-bitnet quantize fixtures/sample_layer_weights.json --output layer0.bnet

# 3. Execute full inference forward pass (RMSNorm + Absmax INT8 + BitLinear)
bin/lh-bitnet infer fixtures/sample_layer_weights.json

# 4. Benchmark SIMD vs Scalar BitLinear throughput
bin/lh-bitnet benchmark --rows 512 --cols 512 --iterations 100

# 5. General information
bin/lh-bitnet --version
bin/lh-bitnet --help
```

### Deterministic Exit Codes

- `0`: Scan passed successfully (Posture Score: `100/100`) or valid forward pass completed.
- `1`: Security posture violations detected (Posture Score: `< 100/100`) or weight validation failure.
- `2`: Command-line usage error, missing arguments, or file I/O error.

---

## Corporate Governance & Attribution

- **Organization**: Logihertz Systems (OPC) Private Limited
- **Creator, Founder, Director & Sole Developer**: **Sameer Tawade** (`ops@logihertz.com` / `security@logihertz.com`)
- **Website**: [https://logihertz.com/](https://logihertz.com/)
- **License**: Apache-2.0
