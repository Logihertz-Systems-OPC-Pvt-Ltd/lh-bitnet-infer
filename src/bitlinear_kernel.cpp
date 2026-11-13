// ==============================================================================
// Product:       Logihertz BitNet 1.58-Bit Inference Engine (lh-bitnet-infer)
// Module:        src/bitlinear_kernel.cpp
// Role:          Multiplication-Free BitLinear Kernel, RMSNorm & KV-Cache Guard
// Organization:  Logihertz Systems (OPC) Private Limited
// Developer:     Sameer Tawade (Creator, Founder, Director & Sole Developer)
// Contact:       ops@logihertz.com / security@logihertz.com
// License:       Apache-2.0
// ==============================================================================

#include "bitlinear_kernel.hpp"
#include "ternary_quantizer.hpp"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>

#if defined(__AVX2__)
#include <immintrin.h>
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace logihertz {

const char* BitLinearKernel::simd_architecture() {
#if defined(__AVX2__)
    return "AVX2 (x86_64 256-bit)";
#elif defined(__ARM_NEON)
    return "ARM_NEON (ARM64 128-bit)";
#else
    return "Scalar";
#endif
}

std::vector<float> BitLinearKernel::rmsnorm_forward(
    const std::vector<float>& x,
    const std::vector<float>& g,
    float eps
) {
    size_t k = x.size();
    if (k == 0) {
        return {};
    }

    double sum_sq = 0.0;
    for (size_t i = 0; i < k; ++i) {
        sum_sq += static_cast<double>(x[i]) * static_cast<double>(x[i]);
    }

    float rms = static_cast<float>(std::sqrt(sum_sq / static_cast<double>(k) + static_cast<double>(eps)));
    float inv_rms = 1.0f / rms;

    std::vector<float> out(k);
    bool has_gain = (g.size() == k);

    for (size_t i = 0; i < k; ++i) {
        float normalized = x[i] * inv_rms;
        out[i] = has_gain ? (normalized * g[i]) : normalized;
    }

    return out;
}

void BitLinearKernel::bitlinear_forward_scalar(
    const PackedTernaryMatrix& matrix,
    const QuantizedActivation& act,
    std::vector<int32_t>& out_accumulators,
    std::vector<float>& out_fp
) {
    size_t m = matrix.rows;
    size_t k = matrix.cols;

    out_accumulators.assign(m, 0);
    out_fp.assign(m, 0.0f);

    float rescale = (matrix.gamma * act.eta) / 127.0f;
    const uint8_t* packed = matrix.packed_data.data();
    const int8_t* act_ptr = act.quantized.data();

    for (size_t i = 0; i < m; ++i) {
        int32_t acc = 0;
        size_t row_offset = i * k;

        for (size_t j = 0; j < k; ++j) {
            size_t idx = row_offset + j;
            size_t byte_idx = idx / 4;
            size_t shift = (idx % 4) * 2;

            uint8_t code = (packed[byte_idx] >> shift) & 0x03;
            // 0 -> -1, 1 -> 0, 2 -> +1
            if (code == 2) {
                acc += static_cast<int32_t>(act_ptr[j]);
            } else if (code == 0) {
                acc -= static_cast<int32_t>(act_ptr[j]);
            }
        }

        out_accumulators[i] = acc;
        out_fp[i] = static_cast<float>(acc) * rescale;
    }
}

void BitLinearKernel::bitlinear_forward_simd(
    const PackedTernaryMatrix& matrix,
    const QuantizedActivation& act,
    std::vector<int32_t>& out_accumulators,
    std::vector<float>& out_fp
) {
#if defined(__AVX2__)
    size_t m = matrix.rows;
    size_t k = matrix.cols;

    out_accumulators.assign(m, 0);
    out_fp.assign(m, 0.0f);

    float rescale = (matrix.gamma * act.eta) / 127.0f;
    const uint8_t* packed = matrix.packed_data.data();
    const int8_t* act_ptr = act.quantized.data();

    for (size_t i = 0; i < m; ++i) {
        int32_t acc = 0;
        size_t row_offset = i * k;
        size_t j = 0;

        // Vectorized accumulation: 32 elements per iteration
        __m256i v_acc32 = _mm256_setzero_si256();

        for (; j + 32 <= k; j += 32) {
            alignas(32) int8_t w_buf[32];
            size_t idx_base = row_offset + j;

            // Unpack 8 consecutive bytes into 32 weights
            for (size_t b = 0; b < 8; ++b) {
                uint8_t byte_val = packed[(idx_base / 4) + b];
                w_buf[b * 4 + 0] = static_cast<int8_t>((byte_val & 0x03)) - 1;
                w_buf[b * 4 + 1] = static_cast<int8_t>(((byte_val >> 2) & 0x03)) - 1;
                w_buf[b * 4 + 2] = static_cast<int8_t>(((byte_val >> 4) & 0x03)) - 1;
                w_buf[b * 4 + 3] = static_cast<int8_t>(((byte_val >> 6) & 0x03)) - 1;
            }

            __m256i v_act = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&act_ptr[j]));
            __m256i v_w = _mm256_load_si256(reinterpret_cast<const __m256i*>(w_buf));

            __m128i act_lo = _mm256_castsi256_si128(v_act);
            __m128i act_hi = _mm256_extracti128_si256(v_act, 1);
            __m128i w_lo = _mm256_castsi256_si128(v_w);
            __m128i w_hi = _mm256_extracti128_si256(v_w, 1);

            __m256i act_lo16 = _mm256_cvtepi8_epi16(act_lo);
            __m256i act_hi16 = _mm256_cvtepi8_epi16(act_hi);
            __m256i w_lo16 = _mm256_cvtepi8_epi16(w_lo);
            __m256i w_hi16 = _mm256_cvtepi8_epi16(w_hi);

            __m256i prod_lo = _mm256_madd_epi16(act_lo16, w_lo16);
            __m256i prod_hi = _mm256_madd_epi16(act_hi16, w_hi16);

            v_acc32 = _mm256_add_epi32(v_acc32, _mm256_add_epi32(prod_lo, prod_hi));
        }

        // Horizontal sum of 8 x 32-bit lane accumulators
        alignas(32) int32_t acc_lanes[8];
        _mm256_store_si256(reinterpret_cast<__m256i*>(acc_lanes), v_acc32);
        for (int l = 0; l < 8; ++l) {
            acc += acc_lanes[l];
        }

        // Remainder scalar loop
        for (; j < k; ++j) {
            size_t idx = row_offset + j;
            size_t byte_idx = idx / 4;
            size_t shift = (idx % 4) * 2;

            uint8_t code = (packed[byte_idx] >> shift) & 0x03;
            if (code == 2) {
                acc += static_cast<int32_t>(act_ptr[j]);
            } else if (code == 0) {
                acc -= static_cast<int32_t>(act_ptr[j]);
            }
        }

        out_accumulators[i] = acc;
        out_fp[i] = static_cast<float>(acc) * rescale;
    }
#else
    // Portable scalar execution when AVX2 is not active
    bitlinear_forward_scalar(matrix, act, out_accumulators, out_fp);
#endif
}

ForwardResult BitLinearKernel::forward(
    const PackedTernaryMatrix& matrix,
    const std::vector<float>& x,
    const std::vector<float>& rmsnorm_g
) {
    auto t0 = std::chrono::high_resolution_clock::now();

    // Step 1: Optional Pre-activation RMSNorm
    std::vector<float> norm_x = rmsnorm_g.empty()
        ? x
        : rmsnorm_forward(x, rmsnorm_g);

    // Step 2: Absmax 8-bit activation quantization
    QuantizedActivation qact = TernaryQuantizer::quantize_activation_absmax(norm_x);

    // Step 3: Multiplication-free BitLinear dot products
    ForwardResult res;
    bitlinear_forward_simd(matrix, qact, res.accumulators, res.output);

    auto t1 = std::chrono::high_resolution_clock::now();
    res.latency_us = std::chrono::duration<double, std::micro>(t1 - t0).count();

    double total_ops = 2.0 * static_cast<double>(matrix.rows) * static_cast<double>(matrix.cols);
    res.gops = (res.latency_us > 0.0) ? (total_ops / (res.latency_us * 1e3)) : 0.0;

    return res;
}

// -----------------------------------------------------------------------------
// KVBufferGuard Implementation
// -----------------------------------------------------------------------------
KVBufferGuard::KVBufferGuard(size_t max_tokens, size_t dim)
    : max_tokens_(max_tokens), dim_(dim), current_tokens_(0) {
    capacity_bytes_ = max_tokens_ * dim_ * sizeof(float);
    k_cache_ = static_cast<float*>(std::malloc(capacity_bytes_));
    v_cache_ = static_cast<float*>(std::malloc(capacity_bytes_));
    if (k_cache_) secure_zero_memory(k_cache_, capacity_bytes_);
    if (v_cache_) secure_zero_memory(v_cache_, capacity_bytes_);
}

KVBufferGuard::~KVBufferGuard() {
    reset();
    if (k_cache_) {
        std::free(k_cache_);
        k_cache_ = nullptr;
    }
    if (v_cache_) {
        std::free(v_cache_);
        v_cache_ = nullptr;
    }
}

bool KVBufferGuard::append_token(const float* k_vec, const float* v_vec) {
    if (current_tokens_ >= max_tokens_ || !k_cache_ || !v_cache_) {
        return false;
    }

    size_t offset = current_tokens_ * dim_;
    std::memcpy(&k_cache_[offset], k_vec, dim_ * sizeof(float));
    std::memcpy(&v_cache_[offset], v_vec, dim_ * sizeof(float));
    current_tokens_++;
    return true;
}

void KVBufferGuard::reset() {
    if (k_cache_) secure_zero_memory(k_cache_, capacity_bytes_);
    if (v_cache_) secure_zero_memory(v_cache_, capacity_bytes_);
    current_tokens_ = 0;
}

void KVBufferGuard::secure_zero_memory(void* ptr, size_t bytes) {
    if (!ptr || bytes == 0) return;
    volatile uint8_t* p = static_cast<volatile uint8_t*>(ptr);
    while (bytes--) {
        *p++ = 0;
    }
}

} // namespace logihertz
