# Contributing to Logihertz BitNet Inference Engine

Thank you for contributing to `lh-bitnet-infer` (`bin/lh-bitnet`). This engine is an official core component of the Logihertz Systems 60-tool open-source ecosystem.

## Core Engineering Principles

1. **Strict Zero External Dependencies**:
   `lh-bitnet-infer` has **0 external third-party library dependencies**. All quantization routines, 2-bit packing schemes, SIMD vectorization, RMSNorm math, and SARIF serializers must be written in pure ISO C++17 standard library headers only (`<vector>`, `<cmath>`, `<string>`, `<fstream>`, `<chrono>`, `<iostream>`).

2. **Strict Compiler Compliance**:
   All C++17 code must compile cleanly without warnings or errors under both `g++` and `clang++` with flags:
   ```bash
   -std=c++17 -O3 -Wall -Wextra -Werror -pedantic -Iinclude
   ```

3. **Deterministic Math & SIMD Equivalence**:
   - The SIMD `BitLinear` kernel (AVX2 on x86_64, ARM NEON on ARM64) must produce **100% bit-identical** integer accumulator results ($A_i$) and identical rescaled floating-point values ($y_i$) when compared against the portable scalar reference path.
   - All quantization math (Absmean $\gamma$, Absmax $\eta$) must remain deterministic across hardware architectures.

4. **10-Rule Auditor Integrity**:
   Any additions or updates to the 10 normative Edge LLM posture rules (`LH-BITNET-001` .. `LH-BITNET-010`) must maintain deterministic posture scoring:
   $$S_{\text{posture}} = \max\left(0, 100 - (25\cdot c_{\text{CRIT}} + 10\cdot c_{\text{HIGH}} + 4\cdot c_{\text{MED}} + 1\cdot c_{\text{LOW}})\right)$$

## Local Development Workflow

```bash
# 1. Clean previous build artifacts
make clean

# 2. Compile binary CLI and run full test suite (85+ assertions)
make all test

# 3. Test CLI locally
bin/lh-bitnet --version
bin/lh-bitnet --help
bin/lh-bitnet benchmark --rows 512 --cols 512
```

## Pull Request Guidelines

- Branch names should be descriptive (e.g., `feat/neon-unroll`, `fix/kv-cache-bounds`).
- Commit messages must follow Conventional Commits (`feat:`, `fix:`, `docs:`, `test:`).
- All pull requests must include unit test assertions in `tests/test_runner.cpp`.
- Submissions are reviewed and maintained by Sameer Tawade (`ops@logihertz.com`).
