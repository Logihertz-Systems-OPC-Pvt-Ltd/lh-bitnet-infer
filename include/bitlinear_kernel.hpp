// ==============================================================================
// Product:       Logihertz BitNet 1.58-Bit Inference Engine (lh-bitnet-infer)
// Module:        include/bitlinear_kernel.hpp
// Role:          Multiplication-Free BitLinear Kernel, RMSNorm & KV-Cache Guard
// Organization:  Logihertz Systems (OPC) Private Limited
// Developer:     Sameer Tawade (Creator, Founder, Director & Sole Developer)
// Contact:       ops@logihertz.com / security@logihertz.com
// License:       Apache-2.0
// ==============================================================================

#ifndef LH_BITLINEAR_KERNEL_HPP
#define LH_BITLINEAR_KERNEL_HPP

#include "bitnet_types.hpp"
#include <cstddef>
#include <cstdint>
#include <vector>

namespace logihertz {

class BitLinearKernel {
public:
    /// Pre-activation Root-Mean-Square Normalization: RMSNorm(x) = (x / RMS(x)) * g
    static std::vector<float> rmsnorm_forward(
        const std::vector<float>& x,
        const std::vector<float>& g,
        float eps = 1e-5f
    );

    /// Portable bit-exact scalar multiplication-free ternary matrix-vector dot product
    static void bitlinear_forward_scalar(
        const PackedTernaryMatrix& matrix,
        const QuantizedActivation& act,
        std::vector<int32_t>& out_accumulators,
        std::vector<float>& out_fp
    );

    /// SIMD-accelerated multiplication-free ternary matrix-vector dot product (AVX2 / NEON)
    static void bitlinear_forward_simd(
        const PackedTernaryMatrix& matrix,
        const QuantizedActivation& act,
        std::vector<int32_t>& out_accumulators,
        std::vector<float>& out_fp
    );

    /// Complete BitLinear forward pass (RMSNorm -> Quantize Act -> SIMD Kernel -> Rescale)
    static ForwardResult forward(
        const PackedTernaryMatrix& matrix,
        const std::vector<float>& x,
        const std::vector<float>& rmsnorm_g = {}
    );

    /// Return active SIMD architecture identifier
    static const char* simd_architecture();
};

/// Bounded KV-Cache Buffer Guard protecting against memory exhaustion & data remanence
class KVBufferGuard {
public:
    explicit KVBufferGuard(size_t max_tokens = 4096, size_t dim = 512);
    ~KVBufferGuard();

    bool append_token(const float* k_vec, const float* v_vec);
    void reset();
    size_t current_tokens() const { return current_tokens_; }
    size_t max_tokens() const { return max_tokens_; }
    size_t memory_bytes() const { return capacity_bytes_; }

private:
    size_t max_tokens_{0};
    size_t dim_{0};
    size_t current_tokens_{0};
    size_t capacity_bytes_{0};
    float* k_cache_{nullptr};
    float* v_cache_{nullptr};

    void secure_zero_memory(void* ptr, size_t bytes);
};

} // namespace logihertz

#endif // LH_BITLINEAR_KERNEL_HPP
