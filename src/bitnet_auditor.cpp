// ==============================================================================
// Product:       Logihertz BitNet 1.58-Bit Inference Engine (lh-bitnet-infer)
// Module:        src/bitnet_auditor.cpp
// Role:          10-Rule Deterministic Edge LLM Posture Auditor Implementation
// Organization:  Logihertz Systems (OPC) Private Limited
// Developer:     Sameer Tawade (Creator, Founder, Director & Sole Developer)
// Contact:       ops@logihertz.com / security@logihertz.com
// License:       Apache-2.0
// ==============================================================================

#include "bitnet_auditor.hpp"
#include <algorithm>

namespace logihertz {

const std::vector<RuleDefinition>& BitNetAuditor::get_rules() {
    static const std::vector<RuleDefinition> s_rules = {
        {
            "LH-BITNET-001",
            "Unverified or Unsigned Ternary Model Weights",
            Severity::Critical,
            "CWE-353",
            25,
            "Model weights lack verified SHA-256 integrity checksum or cryptographic signature.",
            "Set verify_weight_sha256: true and ensure model_signature_verified: true."
        },
        {
            "LH-BITNET-002",
            "Unbounded KV-Cache Memory Allocation",
            Severity::Critical,
            "CWE-770",
            25,
            "Maximum context tokens exceeds 8192, KV-cache exceeds 1024 MB, or OOM guard is disabled.",
            "Constrain max_context_tokens <= 8192, kv_cache_max_mb <= 1024, and enable oom_guard_enabled: true."
        },
        {
            "LH-BITNET-003",
            "Plaintext or Unauthenticated Inference RPC Server",
            Severity::Critical,
            "CWE-319",
            25,
            "Inference RPC server binds to non-loopback interface without TLS or API token authentication.",
            "Bind strictly to 127.0.0.1 or enforce tls_enabled: true and api_auth_required: true."
        },
        {
            "LH-BITNET-004",
            "Excessive Quantization Distortion or Invalid Scale",
            Severity::High,
            "CWE-682",
            10,
            "Quantization MSE exceeds 0.05, perplexity degradation exceeds 5.0%, or absmean_scale_gamma <= 0.0.",
            "Keep quantization_mse_threshold <= 0.05, perplexity_degradation_pct <= 5.0, and absmean_scale_gamma > 0.0."
        },
        {
            "LH-BITNET-005",
            "Missing INT32 Accumulator Guard or Disabled INT8 Clipping",
            Severity::High,
            "CWE-190",
            10,
            "Signed 32-bit accumulator overflow guard is disabled or dynamic activation clipping is bypassed.",
            "Enable int8_activation_clipping: true and int32_accumulator_guard: true."
        },
        {
            "LH-BITNET-006",
            "Unaligned SIMD Tensors or Unchecked 2-Bit Unpacking",
            Severity::High,
            "CWE-119",
            10,
            "SIMD alignment is less than 32 bytes or 2-bit weight unpacking lacks strict bounds checking.",
            "Set simd_alignment_bytes >= 32 (AVX2/NEON) and bounds_checked_unpacking: true."
        },
        {
            "LH-BITNET-007",
            "Missing Prompt Sanitization or Unbounded Token Generation",
            Severity::High,
            "CWE-74",
            10,
            "Maximum generated tokens exceeds 2048, prompt sanitizer is disabled, or stop sequence is unconstrained.",
            "Limit max_new_tokens <= 2048, enable prompt_token_sanitizer_enabled: true, and stop_sequence_enforced: true."
        },
        {
            "LH-BITNET-008",
            "Invalid RMSNorm Epsilon or Disabled Pre-Normalization",
            Severity::Medium,
            "CWE-682",
            4,
            "Pre-activation RMSNorm is disabled or epsilon parameter is outside safe range (0.0, 0.001].",
            "Enable pre_rmsnorm_enabled: true and calibrate rmsnorm_eps within (0.0, 0.001]."
        },
        {
            "LH-BITNET-009",
            "Residual Prompt/KV Telemetry Retention or Unlocked Memory",
            Severity::Medium,
            "CWE-226",
            4,
            "Zeroization of KV-cache on session reset is disabled or model weights are unlocked from RAM.",
            "Set zero_kv_cache_on_reset: true and mlock_model_weights: true."
        },
        {
            "LH-BITNET-010",
            "Missing SoC Thermal Throttling Guard or Non-Deterministic Seed",
            Severity::Low,
            "CWE-755",
            1,
            "Hardware thermal throttling guard is disabled or deterministic generation seed is unconfigured.",
            "Enable thermal_throttle_guard: true and deterministic_seed_enabled: true."
        }
    };
    return s_rules;
}

AuditReport BitNetAuditor::audit_policy(const JsonValue& policy, const std::string& manifest_path) {
    AuditReport report;
    report.manifest_path = manifest_path;
    report.total_rules = 10;

    const auto& rules = get_rules();

    // 1. LH-BITNET-001 (CRITICAL, 25)
    bool verify_sha = policy.has_key("verify_weight_sha256") ? policy.get("verify_weight_sha256")->as_bool(false) : false;
    bool sig_verified = policy.has_key("model_signature_verified") ? policy.get("model_signature_verified")->as_bool(false) : false;
    if (!verify_sha || !sig_verified) {
        report.findings.push_back({
            rules[0].id,
            rules[0].name,
            "CRITICAL",
            rules[0].weight,
            rules[0].cwe,
            "verify_weight_sha256=" + std::string(verify_sha ? "true" : "false") +
            ", model_signature_verified=" + std::string(sig_verified ? "true" : "false") + " (cryptographic validation required).",
            rules[0].remediation
        });
    }

    // 2. LH-BITNET-002 (CRITICAL, 25)
    int64_t max_ctx = policy.has_key("max_context_tokens") ? policy.get("max_context_tokens")->as_int64(16384) : 16384;
    int64_t kv_mb = policy.has_key("kv_cache_max_mb") ? policy.get("kv_cache_max_mb")->as_int64(2048) : 2048;
    bool oom_guard = policy.has_key("oom_guard_enabled") ? policy.get("oom_guard_enabled")->as_bool(false) : false;
    if (max_ctx > 8192 || kv_mb > 1024 || !oom_guard) {
        report.findings.push_back({
            rules[1].id,
            rules[1].name,
            "CRITICAL",
            rules[1].weight,
            rules[1].cwe,
            "max_context_tokens=" + std::to_string(max_ctx) + ", kv_cache_max_mb=" + std::to_string(kv_mb) +
            ", oom_guard_enabled=" + std::string(oom_guard ? "true" : "false") + ".",
            rules[1].remediation
        });
    }

    // 3. LH-BITNET-003 (CRITICAL, 25)
    std::string bind_addr = policy.has_key("bind_address") ? policy.get("bind_address")->as_string("") : "";
    bool tls = policy.has_key("tls_enabled") ? policy.get("tls_enabled")->as_bool(false) : false;
    bool auth = policy.has_key("api_auth_required") ? policy.get("api_auth_required")->as_bool(false) : false;
    if (bind_addr != "127.0.0.1" && (!tls || !auth)) {
        report.findings.push_back({
            rules[2].id,
            rules[2].name,
            "CRITICAL",
            rules[2].weight,
            rules[2].cwe,
            "Inference RPC bound to external address '" + bind_addr + "' with tls_enabled=" + std::string(tls ? "true" : "false") +
            ", api_auth_required=" + std::string(auth ? "true" : "false") + ".",
            rules[2].remediation
        });
    }

    // 4. LH-BITNET-004 (HIGH, 10)
    double mse = policy.has_key("quantization_mse_threshold") ? policy.get("quantization_mse_threshold")->as_double(0.1) : 0.1;
    double ppl = policy.has_key("perplexity_degradation_pct") ? policy.get("perplexity_degradation_pct")->as_double(10.0) : 10.0;
    double gamma = policy.has_key("absmean_scale_gamma") ? policy.get("absmean_scale_gamma")->as_double(0.0) : 0.0;
    if (mse > 0.05 || ppl > 5.0 || gamma <= 0.0) {
        report.findings.push_back({
            rules[3].id,
            rules[3].name,
            "HIGH",
            rules[3].weight,
            rules[3].cwe,
            "quantization_mse_threshold=" + std::to_string(mse) + " (> 0.05), perplexity_degradation_pct=" +
            std::to_string(ppl) + "% (> 5%), or absmean_scale_gamma=" + std::to_string(gamma) + " (<= 0).",
            rules[3].remediation
        });
    }

    // 5. LH-BITNET-005 (HIGH, 10)
    bool int8_clip = policy.has_key("int8_activation_clipping") ? policy.get("int8_activation_clipping")->as_bool(false) : false;
    bool acc_guard = policy.has_key("int32_accumulator_guard") ? policy.get("int32_accumulator_guard")->as_bool(false) : false;
    if (!int8_clip || !acc_guard) {
        report.findings.push_back({
            rules[4].id,
            rules[4].name,
            "HIGH",
            rules[4].weight,
            rules[4].cwe,
            "int8_activation_clipping=" + std::string(int8_clip ? "true" : "false") +
            ", int32_accumulator_guard=" + std::string(acc_guard ? "true" : "false") + ".",
            rules[4].remediation
        });
    }

    // 6. LH-BITNET-006 (HIGH, 10)
    int64_t align_bytes = policy.has_key("simd_alignment_bytes") ? policy.get("simd_alignment_bytes")->as_int64(0) : 0;
    bool bounds_unpack = policy.has_key("bounds_checked_unpacking") ? policy.get("bounds_checked_unpacking")->as_bool(false) : false;
    if (align_bytes < 32 || !bounds_unpack) {
        report.findings.push_back({
            rules[5].id,
            rules[5].name,
            "HIGH",
            rules[5].weight,
            rules[5].cwe,
            "simd_alignment_bytes=" + std::to_string(align_bytes) + " (< 32 bytes) or bounds_checked_unpacking=" +
            std::string(bounds_unpack ? "true" : "false") + ".",
            rules[5].remediation
        });
    }

    // 7. LH-BITNET-007 (HIGH, 10)
    int64_t max_new = policy.has_key("max_new_tokens") ? policy.get("max_new_tokens")->as_int64(4096) : 4096;
    bool sanitizer = policy.has_key("prompt_token_sanitizer_enabled") ? policy.get("prompt_token_sanitizer_enabled")->as_bool(false) : false;
    bool stop_seq = policy.has_key("stop_sequence_enforced") ? policy.get("stop_sequence_enforced")->as_bool(false) : false;
    if (max_new > 2048 || !sanitizer || !stop_seq) {
        report.findings.push_back({
            rules[6].id,
            rules[6].name,
            "HIGH",
            rules[6].weight,
            rules[6].cwe,
            "max_new_tokens=" + std::to_string(max_new) + " (> 2048), prompt_token_sanitizer_enabled=" +
            std::string(sanitizer ? "true" : "false") + ", stop_sequence_enforced=" + std::string(stop_seq ? "true" : "false") + ".",
            rules[6].remediation
        });
    }

    // 8. LH-BITNET-008 (MEDIUM, 4)
    bool pre_rms = policy.has_key("pre_rmsnorm_enabled") ? policy.get("pre_rmsnorm_enabled")->as_bool(false) : false;
    double rms_eps = policy.has_key("rmsnorm_eps") ? policy.get("rmsnorm_eps")->as_double(0.0) : 0.0;
    if (!pre_rms || rms_eps <= 0.0 || rms_eps > 0.001) {
        report.findings.push_back({
            rules[7].id,
            rules[7].name,
            "MEDIUM",
            rules[7].weight,
            rules[7].cwe,
            "pre_rmsnorm_enabled=" + std::string(pre_rms ? "true" : "false") + ", rmsnorm_eps=" +
            std::to_string(rms_eps) + " (must be within (0.0, 0.001]).",
            rules[7].remediation
        });
    }

    // 9. LH-BITNET-009 (MEDIUM, 4)
    bool zero_reset = policy.has_key("zero_kv_cache_on_reset") ? policy.get("zero_kv_cache_on_reset")->as_bool(false) : false;
    bool mlock_weights = policy.has_key("mlock_model_weights") ? policy.get("mlock_model_weights")->as_bool(false) : false;
    if (!zero_reset || !mlock_weights) {
        report.findings.push_back({
            rules[8].id,
            rules[8].name,
            "MEDIUM",
            rules[8].weight,
            rules[8].cwe,
            "zero_kv_cache_on_reset=" + std::string(zero_reset ? "true" : "false") +
            ", mlock_model_weights=" + std::string(mlock_weights ? "true" : "false") + ".",
            rules[8].remediation
        });
    }

    // 10. LH-BITNET-010 (LOW, 1)
    bool thermal = policy.has_key("thermal_throttle_guard") ? policy.get("thermal_throttle_guard")->as_bool(false) : false;
    bool deterministic = policy.has_key("deterministic_seed_enabled") ? policy.get("deterministic_seed_enabled")->as_bool(false) : false;
    if (!thermal || !deterministic) {
        report.findings.push_back({
            rules[9].id,
            rules[9].name,
            "LOW",
            rules[9].weight,
            rules[9].cwe,
            "thermal_throttle_guard=" + std::string(thermal ? "true" : "false") +
            ", deterministic_seed_enabled=" + std::string(deterministic ? "true" : "false") + ".",
            rules[9].remediation
        });
    }

    // Posture score math
    uint32_t total_deduction = 0;
    for (const auto& f : report.findings) {
        total_deduction += f.weight;
        if (f.severity == "CRITICAL") report.critical_count++;
        else if (f.severity == "HIGH") report.high_count++;
        else if (f.severity == "MEDIUM") report.medium_count++;
        else if (f.severity == "LOW") report.low_count++;
    }

    report.score = (total_deduction >= 100) ? 0 : (100 - total_deduction);
    report.passed_rules = report.total_rules - report.findings.size();
    report.status = (report.score == 100) ? "PASSED" : "FAILED";

    return report;
}

} // namespace logihertz
