// ==============================================================================
// Product:       Logihertz BitNet 1.58-Bit Inference Engine (lh-bitnet-infer)
// Module:        tests/test_runner.cpp
// Role:          Deterministic C++17 Test Suite (85+ Assertions)
// Organization:  Logihertz Systems (OPC) Private Limited
// Developer:     Sameer Tawade (Creator, Founder, Director & Sole Developer)
// Contact:       ops@logihertz.com / security@logihertz.com
// License:       Apache-2.0
// ==============================================================================

#include "bitnet_types.hpp"
#include "bitnet_auditor.hpp"
#include "bitlinear_kernel.hpp"
#include "ternary_quantizer.hpp"
#include "json_parser.hpp"
#include "exporters.hpp"

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <cassert>
#include <fstream>
#include <sstream>

using namespace logihertz;

static int g_tests_run = 0;
static int g_tests_passed = 0;
static int g_tests_failed = 0;

#define LH_ASSERT(cond, msg) do { \
    g_tests_run++; \
    if (cond) { \
        g_tests_passed++; \
    } else { \
        g_tests_failed++; \
        std::cerr << "FAIL [" << __FILE__ << ":" << __LINE__ << "]: " << (msg) << "\n"; \
    } \
} while(0)

#define LH_ASSERT_NEAR(a, b, eps, msg) do { \
    g_tests_run++; \
    if (std::fabs((a) - (b)) <= (eps)) { \
        g_tests_passed++; \
    } else { \
        g_tests_failed++; \
        std::cerr << "FAIL [" << __FILE__ << ":" << __LINE__ << "]: " << (msg) \
                  << " (expected " << (b) << ", got " << (a) << ", diff " << std::fabs((a) - (b)) << ")\n"; \
    } \
} while(0)

// -----------------------------------------------------------------------------
// Test Suite 1: Ternary Quantizer & Packing
// -----------------------------------------------------------------------------
static void test_ternary_quantizer() {
    std::cout << "[TEST] Running Ternary Quantizer & Packing tests...\n";

    // 1. Absmean scaling
    std::vector<float> w1 = {1.0f, -1.0f, 0.5f, -0.5f};
    float gamma = TernaryQuantizer::compute_absmean(w1, 2, 2);
    LH_ASSERT_NEAR(gamma, 0.75f, 1e-4f, "Absmean of {1.0, -1.0, 0.5, -0.5} must be ~0.75");

    std::vector<float> w_zero = {0.0f, 0.0f, 0.0f, 0.0f};
    float gamma_zero = TernaryQuantizer::compute_absmean(w_zero, 2, 2);
    LH_ASSERT(gamma_zero > 0.0f, "Absmean of zeroes must have positive epsilon guard");

    // 2. Quantize ternary
    // With gamma = 0.75, threshold is 0.375
    // 1.0 > 0.375 -> +1
    // -1.0 < -0.375 -> -1
    // 0.5 > 0.375 -> +1
    // -0.5 < -0.375 -> -1
    // 0.1 in [-0.375, 0.375] -> 0
    std::vector<float> w2 = {1.0f, -1.0f, 0.1f, -0.2f};
    auto t_quant = TernaryQuantizer::quantize_ternary(w2, gamma);
    LH_ASSERT(t_quant.size() == 4, "Ternary quant size must be 4");
    LH_ASSERT(t_quant[0] == 1, "Weight 1.0 must quantize to +1");
    LH_ASSERT(t_quant[1] == -1, "Weight -1.0 must quantize to -1");
    LH_ASSERT(t_quant[2] == 0, "Weight 0.1 must quantize to 0");
    LH_ASSERT(t_quant[3] == 0, "Weight -0.2 must quantize to 0");

    // 3. 2-bit packing & unpacking roundtrip
    std::vector<int8_t> original_ternary = {
        -1,  0,  1,  0,
         1,  1, -1, -1,
         0,  0,  0,  1,
        -1,  1, -1,  0
    };
    auto packed = TernaryQuantizer::pack_ternary_weights(original_ternary);
    LH_ASSERT(packed.size() == 4, "16 weights must pack into 4 bytes (4 per byte)");

    // Byte 0: {-1, 0, 1, 0} -> codes {0, 1, 2, 1}
    // byte = 0 | (1 << 2) | (2 << 4) | (1 << 6) = 0 + 4 + 32 + 64 = 100
    LH_ASSERT(packed[0] == 100, "Byte 0 packing code mismatch");

    auto unpacked = TernaryQuantizer::unpack_ternary_weights(packed.data(), original_ternary.size());
    LH_ASSERT(unpacked.size() == original_ternary.size(), "Unpacked size must match original");
    bool match = true;
    for (size_t i = 0; i < original_ternary.size(); ++i) {
        if (unpacked[i] != original_ternary[i]) {
            match = false;
            break;
        }
    }
    LH_ASSERT(match, "Unpacked ternary weights must match original bit-for-bit");

    // 4. Absmax activation quantization
    std::vector<float> act = {2.0f, -1.0f, 0.0f, -2.0f};
    auto qact = TernaryQuantizer::quantize_activation_absmax(act);
    LH_ASSERT_NEAR(qact.eta, 2.0f, 1e-4f, "Absmax eta for {2, -1, 0, -2} must be 2.0");
    LH_ASSERT(qact.quantized.size() == 4, "Quantized activation size must be 4");
    LH_ASSERT(qact.quantized[0] == 127, "Max positive activation must quantize to 127");
    LH_ASSERT(qact.quantized[3] == -127, "Max negative activation must quantize to -127");

    // 5. Matrix quantization end-to-end
    std::vector<float> mat_w(64, 0.8f);
    mat_w[10] = -0.9f;
    mat_w[20] = 0.05f;
    auto mat = TernaryQuantizer::quantize_matrix(mat_w, 8, 8);
    LH_ASSERT(mat.rows == 8, "Matrix rows must be 8");
    LH_ASSERT(mat.cols == 8, "Matrix cols must be 8");
    LH_ASSERT(mat.packed_data.size() == 16, "64 elements must pack into 16 bytes");
    LH_ASSERT(mat.gamma > 0.0f, "Matrix gamma must be > 0");
    LH_ASSERT(mat.sha256_hex.length() == 64, "SHA-256 digest length must be 64 hex characters");

    // 6. SHA-256 known digest
    // Empty string SHA-256: e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
    std::string sha_empty = TernaryQuantizer::compute_sha256(reinterpret_cast<const uint8_t*>(""), 0);
    LH_ASSERT(sha_empty == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
              "SHA-256 of empty string must match NIST standard");

    // 7. Binary .bnet file serialization & deserialization
    std::string bnet_path = "/tmp/test_layer.bnet";
    bool save_ok = TernaryQuantizer::save_bnet_file(bnet_path, mat);
    LH_ASSERT(save_ok, "save_bnet_file must succeed");

    PackedTernaryMatrix loaded_mat;
    bool load_ok = TernaryQuantizer::load_bnet_file(bnet_path, loaded_mat);
    LH_ASSERT(load_ok, "load_bnet_file must succeed");
    LH_ASSERT(loaded_mat.rows == mat.rows, "Loaded rows must match");
    LH_ASSERT(loaded_mat.cols == mat.cols, "Loaded cols must match");
    LH_ASSERT_NEAR(loaded_mat.gamma, mat.gamma, 1e-5f, "Loaded gamma must match");
    LH_ASSERT(loaded_mat.sha256_hex == mat.sha256_hex, "Loaded SHA-256 must match");
    LH_ASSERT(loaded_mat.packed_data == mat.packed_data, "Loaded packed data must match identically");
}

// -----------------------------------------------------------------------------
// Test Suite 2: BitLinear Kernel & SIMD Bit-Exactness
// -----------------------------------------------------------------------------
static void test_bitlinear_kernel() {
    std::cout << "[TEST] Running BitLinear Kernel & SIMD tests...\n";

    // 1. RMSNorm
    std::vector<float> x = {2.0f, -2.0f, 2.0f, -2.0f};
    std::vector<float> g = {1.0f, 1.0f, 1.0f, 1.0f};
    auto x_norm = BitLinearKernel::rmsnorm_forward(x, g, 1e-5f);
    LH_ASSERT(x_norm.size() == 4, "RMSNorm output size must match input");
    // RMS = sqrt( (4+4+4+4)/4 ) = 2.0. So x / RMS = {1, -1, 1, -1}
    LH_ASSERT_NEAR(x_norm[0], 1.0f, 1e-3f, "RMSNorm x[0] must be 1.0");
    LH_ASSERT_NEAR(x_norm[1], -1.0f, 1e-3f, "RMSNorm x[1] must be -1.0");

    // 2. Multiplication-Free Accumulation (Scalar vs SIMD Bit-Exactness)
    // 32 rows x 32 cols
    size_t M = 32;
    size_t K = 32;
    std::vector<float> weights(M * K);
    for (size_t i = 0; i < M * K; ++i) {
        weights[i] = std::sin(static_cast<float>(i)) * 1.5f;
    }
    PackedTernaryMatrix pmat = TernaryQuantizer::quantize_matrix(weights, M, K);

    std::vector<float> act(K);
    for (size_t j = 0; j < K; ++j) {
        act[j] = std::cos(static_cast<float>(j)) * 2.0f;
    }
    QuantizedActivation qact = TernaryQuantizer::quantize_activation_absmax(act);

    std::vector<int32_t> scalar_acc(M, 0);
    std::vector<float> scalar_fp(M, 0.0f);
    BitLinearKernel::bitlinear_forward_scalar(pmat, qact, scalar_acc, scalar_fp);

    std::vector<int32_t> simd_acc(M, 0);
    std::vector<float> simd_fp(M, 0.0f);
    BitLinearKernel::bitlinear_forward_simd(pmat, qact, simd_acc, simd_fp);

    bool bit_exact = true;
    for (size_t i = 0; i < M; ++i) {
        if (scalar_acc[i] != simd_acc[i]) {
            bit_exact = false;
            break;
        }
    }
    LH_ASSERT(bit_exact, "SIMD accumulation must match Scalar accumulation with 100% bit-exactness");

    // 3. Full BitLinear forward pass
    ForwardResult res = BitLinearKernel::forward(pmat, act, g);
    LH_ASSERT(res.output.size() == M, "Forward output size must match rows M");
    LH_ASSERT(res.accumulators.size() == M, "Accumulators size must match rows M");
    LH_ASSERT(res.latency_us >= 0.0, "Latency must be non-negative");
    LH_ASSERT(res.gops >= 0.0, "Throughput GOPS must be non-negative");

    // 4. KV-Cache Buffer Guard
    KVBufferGuard guard(4, 16);
    LH_ASSERT(guard.max_tokens() == 4, "KV cache max tokens must be 4");
    LH_ASSERT(guard.current_tokens() == 0, "Initial tokens must be 0");

    std::vector<float> k_tok(16, 1.0f);
    std::vector<float> v_tok(16, 2.0f);

    LH_ASSERT(guard.append_token(k_tok.data(), v_tok.data()), "Token 1 append must succeed");
    LH_ASSERT(guard.append_token(k_tok.data(), v_tok.data()), "Token 2 append must succeed");
    LH_ASSERT(guard.append_token(k_tok.data(), v_tok.data()), "Token 3 append must succeed");
    LH_ASSERT(guard.append_token(k_tok.data(), v_tok.data()), "Token 4 append must succeed");
    LH_ASSERT(guard.current_tokens() == 4, "Token count must be 4");

    // Token 5 exceeds max_tokens -> buffer guard prevents OOM
    LH_ASSERT(!guard.append_token(k_tok.data(), v_tok.data()), "OOM guard must block appending beyond max_tokens");

    // Reset and zeroize
    guard.reset();
    LH_ASSERT(guard.current_tokens() == 0, "Tokens must be 0 after reset");
}

// -----------------------------------------------------------------------------
// Test Suite 3: Clean-Room JSON Parser
// -----------------------------------------------------------------------------
static void test_json_parser() {
    std::cout << "[TEST] Running JSON Parser tests...\n";

    std::string json_str = R"({
        "name": "BitNet-1.58b",
        "layers": 32,
        "gamma": 0.825,
        "active": true,
        "disabled": false,
        "extra": null,
        "tags": ["edge", "simd", "ternary"],
        "sub": {
            "key": "val"
        }
    })";

    std::string err;
    JsonValue root = JsonValue::parse(json_str, &err);
    LH_ASSERT(err.empty(), "Valid JSON parse must succeed without error");
    LH_ASSERT(root.is_object(), "Root must be an object");
    LH_ASSERT(root.has_key("name"), "Must have key 'name'");
    LH_ASSERT(root.get("name")->as_string() == "BitNet-1.58b", "Key 'name' value mismatch");
    LH_ASSERT(root.get("layers")->as_int64() == 32, "Key 'layers' value mismatch");
    LH_ASSERT_NEAR(root.get("gamma")->as_double(), 0.825, 1e-4, "Key 'gamma' value mismatch");
    LH_ASSERT(root.get("active")->as_bool() == true, "Key 'active' must be true");
    LH_ASSERT(root.get("disabled")->as_bool() == false, "Key 'disabled' must be false");
    LH_ASSERT(root.get("extra")->is_null(), "Key 'extra' must be null");

    const JsonValue* tags = root.get("tags");
    LH_ASSERT(tags && tags->is_array(), "Key 'tags' must be an array");
    LH_ASSERT(tags->arr_val.size() == 3, "'tags' array must have 3 elements");
    LH_ASSERT(tags->arr_val[0].as_string() == "edge", "Tag 0 mismatch");
    LH_ASSERT(tags->arr_val[1].as_string() == "simd", "Tag 1 mismatch");
    LH_ASSERT(tags->arr_val[2].as_string() == "ternary", "Tag 2 mismatch");

    // Malformed JSON check
    std::string bad_json = "{ name: unquoted }";
    std::string bad_err;
    JsonValue bad_root = JsonValue::parse(bad_json, &bad_err);
    LH_ASSERT(!bad_err.empty(), "Malformed JSON must return non-empty error");
}

// -----------------------------------------------------------------------------
// Test Suite 4: 10-Rule Auditor & Scoring Verification
// -----------------------------------------------------------------------------
static void test_bitnet_auditor() {
    std::cout << "[TEST] Running 10-Rule BitNet Auditor tests...\n";

    // 1. Hardened config test (Score: 100/100, PASSED)
    std::string hard_str;
    std::ifstream hard_f("fixtures/hardened_bitnet_config.json");
    if (hard_f.is_open()) {
        std::stringstream ss;
        ss << hard_f.rdbuf();
        hard_str = ss.str();
    }
    JsonValue hard_cfg = JsonValue::parse(hard_str);
    AuditReport hard_rep = BitNetAuditor::audit_policy(hard_cfg, "fixtures/hardened_bitnet_config.json");
    LH_ASSERT(hard_rep.score == 100, "Hardened config score must be 100");
    LH_ASSERT(hard_rep.status == "PASSED", "Hardened config status must be PASSED");
    LH_ASSERT(hard_rep.findings.empty(), "Hardened config must have 0 findings");
    LH_ASSERT(hard_rep.passed_rules == 10, "Hardened config must pass all 10 rules");

    // 2. Vulnerable config test (Score: 0/100, FAILED, all 10 findings)
    std::string vuln_str;
    std::ifstream vuln_f("fixtures/vulnerable_bitnet_config.json");
    if (vuln_f.is_open()) {
        std::stringstream ss;
        ss << vuln_f.rdbuf();
        vuln_str = ss.str();
    }
    JsonValue vuln_cfg = JsonValue::parse(vuln_str);
    AuditReport vuln_rep = BitNetAuditor::audit_policy(vuln_cfg, "fixtures/vulnerable_bitnet_config.json");
    LH_ASSERT(vuln_rep.score == 0, "Vulnerable config score must be 0");
    LH_ASSERT(vuln_rep.status == "FAILED", "Vulnerable config status must be FAILED");
    LH_ASSERT(vuln_rep.findings.size() == 10, "Vulnerable config must trigger all 10 rules");
    LH_ASSERT(vuln_rep.critical_count == 3, "Vulnerable config must have 3 CRITICAL findings");
    LH_ASSERT(vuln_rep.high_count == 4, "Vulnerable config must have 4 HIGH findings");
    LH_ASSERT(vuln_rep.medium_count == 2, "Vulnerable config must have 2 MEDIUM findings");
    LH_ASSERT(vuln_rep.low_count == 1, "Vulnerable config must have 1 LOW finding");

    // 3. Isolated Single Rule Tests
    // Test LH-BITNET-001 (CRITICAL, deduction 25 -> score 75)
    JsonValue test_r1 = hard_cfg;
    for (auto& p : test_r1.obj_val) {
        if (p.first == "verify_weight_sha256") p.second = JsonValue(false);
    }
    AuditReport rep_r1 = BitNetAuditor::audit_policy(test_r1, "test_r1.json");
    LH_ASSERT(rep_r1.score == 75, "Failing LH-BITNET-001 must yield score 75");
    LH_ASSERT(rep_r1.findings.size() == 1, "Must have exactly 1 finding");
    LH_ASSERT(rep_r1.findings[0].rule_id == "LH-BITNET-001", "Finding must be LH-BITNET-001");

    // Test LH-BITNET-004 (HIGH, deduction 10 -> score 90)
    JsonValue test_r4 = hard_cfg;
    for (auto& p : test_r4.obj_val) {
        if (p.first == "quantization_mse_threshold") p.second = JsonValue(0.12);
    }
    AuditReport rep_r4 = BitNetAuditor::audit_policy(test_r4, "test_r4.json");
    LH_ASSERT(rep_r4.score == 90, "Failing LH-BITNET-004 must yield score 90");
    LH_ASSERT(rep_r4.findings[0].rule_id == "LH-BITNET-004", "Finding must be LH-BITNET-004");

    // Test LH-BITNET-008 (MEDIUM, deduction 4 -> score 96)
    JsonValue test_r8 = hard_cfg;
    for (auto& p : test_r8.obj_val) {
        if (p.first == "pre_rmsnorm_enabled") p.second = JsonValue(false);
    }
    AuditReport rep_r8 = BitNetAuditor::audit_policy(test_r8, "test_r8.json");
    LH_ASSERT(rep_r8.score == 96, "Failing LH-BITNET-008 must yield score 96");
    LH_ASSERT(rep_r8.findings[0].rule_id == "LH-BITNET-008", "Finding must be LH-BITNET-008");

    // Test LH-BITNET-010 (LOW, deduction 1 -> score 99)
    JsonValue test_r10 = hard_cfg;
    for (auto& p : test_r10.obj_val) {
        if (p.first == "thermal_throttle_guard") p.second = JsonValue(false);
    }
    AuditReport rep_r10 = BitNetAuditor::audit_policy(test_r10, "test_r10.json");
    LH_ASSERT(rep_r10.score == 99, "Failing LH-BITNET-010 must yield score 99");
    LH_ASSERT(rep_r10.findings[0].rule_id == "LH-BITNET-010", "Finding must be LH-BITNET-010");
}

// -----------------------------------------------------------------------------
// Test Suite 5: Exporters (Table, Canonical JSON, SARIF v2.1.0)
// -----------------------------------------------------------------------------
static void test_exporters() {
    std::cout << "[TEST] Running Exporter tests...\n";

    std::string vuln_str;
    std::ifstream vuln_f("fixtures/vulnerable_bitnet_config.json");
    if (vuln_f.is_open()) {
        std::stringstream ss;
        ss << vuln_f.rdbuf();
        vuln_str = ss.str();
    }
    JsonValue vuln_cfg = JsonValue::parse(vuln_str);
    AuditReport report = BitNetAuditor::audit_policy(vuln_cfg, "fixtures/vulnerable_bitnet_config.json");

    // 1. Table
    std::string tbl = Exporters::format_table(report);
    LH_ASSERT(tbl.find("Logihertz BitNet") != std::string::npos, "Table must contain banner");
    LH_ASSERT(tbl.find("LH-BITNET-001") != std::string::npos, "Table must contain rule findings");
    LH_ASSERT(tbl.find("FAILED") != std::string::npos, "Table must reflect FAILED status");

    // 2. Canonical JSON
    std::string j_out = Exporters::format_json(report);
    std::string j_err;
    JsonValue j_parsed = JsonValue::parse(j_out, &j_err);
    LH_ASSERT(j_err.empty(), "Exported JSON must be valid JSON");
    LH_ASSERT(j_parsed.get("score")->as_int64() == 0, "Exported JSON score must match report");
    LH_ASSERT(j_parsed.get("findings")->arr_val.size() == 10, "Exported JSON must have 10 findings");

    // 3. SARIF v2.1.0
    std::string s_out = Exporters::format_sarif(report);
    std::string s_err;
    JsonValue s_parsed = JsonValue::parse(s_out, &s_err);
    LH_ASSERT(s_err.empty(), "Exported SARIF must be valid JSON");
    LH_ASSERT(s_parsed.has_key("$schema"), "SARIF must have '$schema'");
    LH_ASSERT(s_parsed.has_key("runs"), "SARIF must have 'runs'");
    const JsonValue* runs = s_parsed.get("runs");
    LH_ASSERT(runs && runs->arr_val.size() == 1, "SARIF must contain 1 run");
    const JsonValue* results = runs->arr_val[0].get("results");
    LH_ASSERT(results && results->arr_val.size() == 10, "SARIF must have 10 results");
}

// -----------------------------------------------------------------------------
// Main Test Runner Entrypoint
// -----------------------------------------------------------------------------
int main() {
    std::cout << "======================================================================\n"
              << "       LOGIHERTZ BITNET INFERENCE ENGINE C++17 TEST SUITE\n"
              << "======================================================================\n";

    test_ternary_quantizer();
    test_bitlinear_kernel();
    test_json_parser();
    test_bitnet_auditor();
    test_exporters();

    std::cout << "======================================================================\n"
              << " TEST SUMMARY: " << g_tests_run << " assertions checked.\n"
              << " PASSED:       " << g_tests_passed << "\n"
              << " FAILED:       " << g_tests_failed << "\n"
              << "======================================================================\n";

    return (g_tests_failed == 0) ? 0 : 1;
}
