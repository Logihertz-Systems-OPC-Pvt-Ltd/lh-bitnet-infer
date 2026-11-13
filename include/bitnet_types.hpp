// ==============================================================================
// Product:       Logihertz BitNet 1.58-Bit Inference Engine (lh-bitnet-infer)
// Module:        include/bitnet_types.hpp
// Role:          Core Data Types, Packed Tensor Layouts, and Auditor Models
// Organization:  Logihertz Systems (OPC) Private Limited
// Developer:     Sameer Tawade (Creator, Founder, Director & Sole Developer)
// Contact:       ops@logihertz.com / security@logihertz.com
// License:       Apache-2.0
// ==============================================================================

#ifndef LH_BITNET_TYPES_HPP
#define LH_BITNET_TYPES_HPP

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace logihertz {

/// Binary magic header for packed BitNet 1.58-bit files ("BNET\x01\x58")
constexpr uint8_t BNET_MAGIC[6] = {'B', 'N', 'E', 'T', 0x01, 0x58};

/// Packed 2-bit ternary weight matrix representation (4 weights per uint8_t byte)
struct PackedTernaryMatrix {
    size_t rows{0};                     ///< Output dimension M
    size_t cols{0};                     ///< Input dimension K
    float gamma{1.0f};                  ///< Absmean quantization scale factor
    std::vector<uint8_t> packed_data;   ///< Packed byte array: ceil(M * K / 4) bytes
    std::string sha256_hex;             ///< FIPS 180-4 SHA-256 weight checksum
    float sparsity_ratio{0.0f};         ///< Fraction of ternary zero weights
    float mse{0.0f};                    ///< Quantization Mean Squared Error
};

/// High-level layer container holding full precision weights and RMSNorm gains
struct LayerWeights {
    size_t rows{0};
    size_t cols{0};
    std::vector<float> weights;          ///< M * K continuous FP32 weights
    std::vector<float> rmsnorm_gain;     ///< K-dimensional RMSNorm scale vector g
    std::vector<float> input_activation; ///< K-dimensional test input activation x
};

/// Dynamically quantized 8-bit activation vector
struct QuantizedActivation {
    std::vector<int8_t> quantized;      ///< Clamped signed 8-bit activations [-127, +127]
    float eta{1.0f};                     ///< Maximum magnitude scale factor
};

/// Result of BitLinear forward inference pass
struct ForwardResult {
    std::vector<float> output;           ///< Final rescaled FP32 activations
    std::vector<int32_t> accumulators;   ///< Raw 32-bit integer accumulation sums
    double latency_us{0.0};              ///< Kernel compute latency in microseconds
    double gops{0.0};                    ///< Giga Operations Per Second throughput
};

/// Security classification severity
enum class Severity {
    Critical,
    High,
    Medium,
    Low
};

inline const char* severity_to_string(Severity s) {
    switch (s) {
        case Severity::Critical: return "CRITICAL";
        case Severity::High:     return "HIGH";
        case Severity::Medium:   return "MEDIUM";
        case Severity::Low:      return "LOW";
    }
    return "UNKNOWN";
}

inline uint32_t severity_weight(Severity s) {
    switch (s) {
        case Severity::Critical: return 25;
        case Severity::High:     return 10;
        case Severity::Medium:   return 4;
        case Severity::Low:      return 1;
    }
    return 0;
}

/// Normative security rule specification
struct RuleDefinition {
    std::string id;
    std::string name;
    Severity severity;
    std::string cwe;
    uint32_t weight;
    std::string description;
    std::string remediation;
};

/// Individual detected posture violation
struct Finding {
    std::string rule_id;
    std::string rule_name;
    std::string severity;
    uint32_t weight{0};
    std::string cwe;
    std::string message;
    std::string remediation;
};

/// Comprehensive Edge LLM posture evaluation report
struct AuditReport {
    std::string manifest_path;
    uint32_t score{100};
    std::string status{"PASSED"};
    size_t total_rules{10};
    size_t passed_rules{10};
    size_t critical_count{0};
    size_t high_count{0};
    size_t medium_count{0};
    size_t low_count{0};
    std::vector<Finding> findings;
};

} // namespace logihertz

#endif // LH_BITNET_TYPES_HPP
