// ==============================================================================
// Product:       Logihertz BitNet 1.58-Bit Inference Engine (lh-bitnet-infer)
// Module:        include/ternary_quantizer.hpp
// Role:          1.58-Bit Absmean Weight Quantizer, 2-Bit Packer & Activation Quantizer
// Organization:  Logihertz Systems (OPC) Private Limited
// Developer:     Sameer Tawade (Creator, Founder, Director & Sole Developer)
// Contact:       ops@logihertz.com / security@logihertz.com
// License:       Apache-2.0
// ==============================================================================

#ifndef LH_TERNARY_QUANTIZER_HPP
#define LH_TERNARY_QUANTIZER_HPP

#include "bitnet_types.hpp"
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace logihertz {

class TernaryQuantizer {
public:
    /// Calculate Absmean dynamic scaling factor: gamma = (1 / MK) * sum(|W_ij|) + 1e-8
    static float compute_absmean(const std::vector<float>& weights, size_t rows, size_t cols);

    /// Quantize continuous FP32 weights into ternary {-1, 0, +1}
    static std::vector<int8_t> quantize_ternary(const std::vector<float>& weights, float gamma);

    /// Pack ternary weights {-1, 0, +1} into 2-bit codes (4 weights per byte)
    static std::vector<uint8_t> pack_ternary_weights(const std::vector<int8_t>& ternary_weights);

    /// Unpack 2-bit codes into ternary weights {-1, 0, +1}
    static std::vector<int8_t> unpack_ternary_weights(
        const uint8_t* packed_data,
        size_t total_weights,
        bool bounds_check = true
    );

    /// 8-bit dynamic per-token activation quantization (Absmax): eta = max(|x_j|) + 1e-8
    static QuantizedActivation quantize_activation_absmax(const std::vector<float>& x);

    /// Complete end-to-end matrix quantization and packing pipeline
    static PackedTernaryMatrix quantize_matrix(
        const std::vector<float>& weights,
        size_t rows,
        size_t cols
    );

    /// Cryptographic FIPS 180-4 SHA-256 weight integrity digest
    static std::string compute_sha256(const uint8_t* data, size_t len);

    /// Serialize packed ternary matrix into binary .bnet format
    static bool save_bnet_file(const std::string& path, const PackedTernaryMatrix& matrix);

    /// Deserialize binary .bnet file into packed ternary matrix
    static bool load_bnet_file(const std::string& path, PackedTernaryMatrix& matrix);
};

} // namespace logihertz

#endif // LH_TERNARY_QUANTIZER_HPP
