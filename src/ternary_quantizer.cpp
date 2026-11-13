// ==============================================================================
// Product:       Logihertz BitNet 1.58-Bit Inference Engine (lh-bitnet-infer)
// Module:        src/ternary_quantizer.cpp
// Role:          Absmean Quantization, 2-Bit Storage Packing & Cryptographic Digest
// Organization:  Logihertz Systems (OPC) Private Limited
// Developer:     Sameer Tawade (Creator, Founder, Director & Sole Developer)
// Contact:       ops@logihertz.com / security@logihertz.com
// License:       Apache-2.0
// ==============================================================================

#include "ternary_quantizer.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace logihertz {

namespace {

// Clean-room FIPS 180-4 SHA-256 implementation
inline uint32_t rotr32(uint32_t val, uint32_t bits) {
    return (val >> bits) | (val << (32 - bits));
}

const uint32_t SHA256_K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

} // namespace

std::string TernaryQuantizer::compute_sha256(const uint8_t* data, size_t len) {
    uint32_t state[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    };

    std::vector<uint8_t> buffer(data, data + len);
    uint64_t total_bits = static_cast<uint64_t>(len) * 8ULL;

    buffer.push_back(0x80);
    while ((buffer.size() % 64) != 56) {
        buffer.push_back(0x00);
    }
    for (int i = 7; i >= 0; --i) {
        buffer.push_back(static_cast<uint8_t>((total_bits >> (i * 8)) & 0xff));
    }

    for (size_t chunk = 0; chunk < buffer.size(); chunk += 64) {
        uint32_t w[64];
        for (int i = 0; i < 16; ++i) {
            w[i] = (static_cast<uint32_t>(buffer[chunk + i * 4]) << 24) |
                   (static_cast<uint32_t>(buffer[chunk + i * 4 + 1]) << 16) |
                   (static_cast<uint32_t>(buffer[chunk + i * 4 + 2]) << 8) |
                   (static_cast<uint32_t>(buffer[chunk + i * 4 + 3]));
        }
        for (int i = 16; i < 64; ++i) {
            uint32_t s0 = rotr32(w[i - 15], 7) ^ rotr32(w[i - 15], 18) ^ (w[i - 15] >> 3);
            uint32_t s1 = rotr32(w[i - 2], 17) ^ rotr32(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }

        uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
        uint32_t e = state[4], f = state[5], g = state[6], h = state[7];

        for (int i = 0; i < 64; ++i) {
            uint32_t s1 = rotr32(e, 6) ^ rotr32(e, 11) ^ rotr32(e, 25);
            uint32_t ch = (e & f) ^ ((~e) & g);
            uint32_t temp1 = h + s1 + ch + SHA256_K[i] + w[i];
            uint32_t s0 = rotr32(a, 2) ^ rotr32(a, 13) ^ rotr32(a, 22);
            uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            uint32_t temp2 = s0 + maj;

            h = g;
            g = f;
            f = e;
            e = d + temp1;
            d = c;
            c = b;
            b = a;
            a = temp1 + temp2;
        }

        state[0] += a; state[1] += b; state[2] += c; state[3] += d;
        state[4] += e; state[5] += f; state[6] += g; state[7] += h;
    }

    std::ostringstream oss;
    for (int i = 0; i < 8; ++i) {
        oss << std::hex << std::setw(8) << std::setfill('0') << state[i];
    }
    return oss.str();
}

float TernaryQuantizer::compute_absmean(const std::vector<float>& weights, size_t rows, size_t cols) {
    size_t count = rows * cols;
    if (count == 0 || weights.empty()) {
        return 1e-8f;
    }

    double sum = 0.0;
    for (size_t i = 0; i < count; ++i) {
        sum += std::abs(weights[i]);
    }
    return static_cast<float>(sum / static_cast<double>(count)) + 1e-8f;
}

std::vector<int8_t> TernaryQuantizer::quantize_ternary(const std::vector<float>& weights, float gamma) {
    std::vector<int8_t> result(weights.size());
    if (gamma <= 0.0f) {
        gamma = 1e-8f;
    }

    for (size_t i = 0; i < weights.size(); ++i) {
        float scaled = std::round(weights[i] / gamma);
        int8_t val = (scaled > 1.0f) ? 1 : ((scaled < -1.0f) ? -1 : static_cast<int8_t>(scaled));
        result[i] = val;
    }
    return result;
}

std::vector<uint8_t> TernaryQuantizer::pack_ternary_weights(const std::vector<int8_t>& ternary_weights) {
    size_t total = ternary_weights.size();
    size_t packed_bytes = (total + 3) / 4;
    std::vector<uint8_t> packed(packed_bytes, 0);

    for (size_t i = 0; i < total; ++i) {
        size_t byte_idx = i / 4;
        size_t shift = (i % 4) * 2;
        int8_t w = ternary_weights[i];
        uint8_t code = static_cast<uint8_t>(w + 1) & 0x03; // -1->0, 0->1, +1->2
        packed[byte_idx] |= (code << shift);
    }
    return packed;
}

std::vector<int8_t> TernaryQuantizer::unpack_ternary_weights(
    const uint8_t* packed_data,
    size_t total_weights,
    bool bounds_check
) {
    std::vector<int8_t> out(total_weights);
    size_t packed_len = (total_weights + 3) / 4;

    for (size_t i = 0; i < total_weights; ++i) {
        size_t byte_idx = i / 4;
        if (bounds_check && byte_idx >= packed_len) {
            out[i] = 0;
            continue;
        }
        size_t shift = (i % 4) * 2;
        uint8_t code = (packed_data[byte_idx] >> shift) & 0x03;
        out[i] = static_cast<int8_t>(code) - 1; // 0->-1, 1->0, 2->+1
    }
    return out;
}

QuantizedActivation TernaryQuantizer::quantize_activation_absmax(const std::vector<float>& x) {
    QuantizedActivation qa;
    qa.quantized.resize(x.size());

    float max_abs = 1e-8f;
    for (float val : x) {
        float a = std::abs(val);
        if (a > max_abs) {
            max_abs = a;
        }
    }
    qa.eta = max_abs;

    float scale = 127.0f / max_abs;
    for (size_t i = 0; i < x.size(); ++i) {
        float scaled = std::round(x[i] * scale);
        float clamped = std::clamp(scaled, -127.0f, 127.0f);
        qa.quantized[i] = static_cast<int8_t>(clamped);
    }

    return qa;
}

PackedTernaryMatrix TernaryQuantizer::quantize_matrix(
    const std::vector<float>& weights,
    size_t rows,
    size_t cols
) {
    PackedTernaryMatrix matrix;
    matrix.rows = rows;
    matrix.cols = cols;

    matrix.gamma = compute_absmean(weights, rows, cols);
    std::vector<int8_t> ternary = quantize_ternary(weights, matrix.gamma);
    matrix.packed_data = pack_ternary_weights(ternary);

    // Compute sparsity and MSE
    size_t zero_count = 0;
    double sum_sq_err = 0.0;
    size_t count = rows * cols;

    for (size_t i = 0; i < count; ++i) {
        if (ternary[i] == 0) {
            zero_count++;
        }
        float approx = static_cast<float>(ternary[i]) * matrix.gamma;
        float diff = weights[i] - approx;
        sum_sq_err += diff * diff;
    }

    matrix.sparsity_ratio = (count > 0) ? (static_cast<float>(zero_count) / static_cast<float>(count)) : 0.0f;
    matrix.mse = (count > 0) ? static_cast<float>(sum_sq_err / static_cast<double>(count)) : 0.0f;

    // Checksum over header fields + packed data
    std::vector<uint8_t> digest_payload;
    digest_payload.reserve(matrix.packed_data.size() + 24);

    uint64_t r_u64 = rows;
    uint64_t c_u64 = cols;
    const uint8_t* r_bytes = reinterpret_cast<const uint8_t*>(&r_u64);
    const uint8_t* c_bytes = reinterpret_cast<const uint8_t*>(&c_u64);
    const uint8_t* g_bytes = reinterpret_cast<const uint8_t*>(&matrix.gamma);

    digest_payload.insert(digest_payload.end(), r_bytes, r_bytes + 8);
    digest_payload.insert(digest_payload.end(), c_bytes, c_bytes + 8);
    digest_payload.insert(digest_payload.end(), g_bytes, g_bytes + 4);
    digest_payload.insert(digest_payload.end(), matrix.packed_data.begin(), matrix.packed_data.end());

    matrix.sha256_hex = compute_sha256(digest_payload.data(), digest_payload.size());

    return matrix;
}

bool TernaryQuantizer::save_bnet_file(const std::string& path, const PackedTernaryMatrix& matrix) {
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    file.write(reinterpret_cast<const char*>(BNET_MAGIC), 6);

    uint32_t r32 = static_cast<uint32_t>(matrix.rows);
    uint32_t c32 = static_cast<uint32_t>(matrix.cols);
    file.write(reinterpret_cast<const char*>(&r32), 4);
    file.write(reinterpret_cast<const char*>(&c32), 4);
    file.write(reinterpret_cast<const char*>(&matrix.gamma), 4);

    uint32_t sha_len = static_cast<uint32_t>(matrix.sha256_hex.size());
    file.write(reinterpret_cast<const char*>(&sha_len), 4);
    file.write(matrix.sha256_hex.data(), sha_len);

    uint64_t data_len = matrix.packed_data.size();
    file.write(reinterpret_cast<const char*>(&data_len), 8);
    file.write(reinterpret_cast<const char*>(matrix.packed_data.data()), data_len);

    return file.good();
}

bool TernaryQuantizer::load_bnet_file(const std::string& path, PackedTernaryMatrix& matrix) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    uint8_t magic[6];
    file.read(reinterpret_cast<char*>(magic), 6);
    if (std::memcmp(magic, BNET_MAGIC, 6) != 0) {
        return false;
    }

    uint32_t r32 = 0, c32 = 0;
    file.read(reinterpret_cast<char*>(&r32), 4);
    file.read(reinterpret_cast<char*>(&c32), 4);
    matrix.rows = r32;
    matrix.cols = c32;
    file.read(reinterpret_cast<char*>(&matrix.gamma), 4);

    uint32_t sha_len = 0;
    file.read(reinterpret_cast<char*>(&sha_len), 4);
    if (sha_len > 128) {
        return false;
    }
    matrix.sha256_hex.resize(sha_len);
    file.read(&matrix.sha256_hex[0], sha_len);

    uint64_t data_len = 0;
    file.read(reinterpret_cast<char*>(&data_len), 8);
    matrix.packed_data.resize(data_len);
    file.read(reinterpret_cast<char*>(matrix.packed_data.data()), data_len);

    return file.good();
}

} // namespace logihertz
