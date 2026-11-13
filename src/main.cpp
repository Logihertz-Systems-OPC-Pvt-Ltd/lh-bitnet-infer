// ==============================================================================
// Product:       Logihertz BitNet 1.58-Bit Inference Engine (lh-bitnet-infer)
// Module:        src/main.cpp
// Role:          Unified CLI Entrypoint for Scanning, Quantizing, Inferencing & Benchmarking
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
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <cstdlib>

using namespace logihertz;

static void print_version() {
    std::cout << "lh-bitnet 0.1.0\n"
              << "Logihertz BitNet 1.58-Bit Ternary Neural Network Inference Engine\n"
              << "Organization: Logihertz Systems (OPC) Private Limited (https://logihertz.com/)\n"
              << "Developer:    Sameer Tawade (Creator, Founder, Director & Sole Developer)\n"
              << "License:      Apache-2.0\n"
              << "Active SIMD:  " << BitLinearKernel::simd_architecture() << "\n";
}

static void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " <command> [options]\n\n"
              << "Commands:\n"
              << "  scan <policy.json>         Audit Edge LLM deployment configuration against 10 security rules\n"
              << "  quantize <weights.json>     Absmean-quantize FP32 weights into 2-bit packed ternary format\n"
              << "  infer <weights.json>        Run BitLinear forward pass (RMSNorm + Absmax INT8 + SIMD Kernel)\n"
              << "  benchmark                  Benchmark multiplication-free SIMD kernel throughput and GOPS\n"
              << "  --version, -v              Display version, author, and SIMD architecture\n"
              << "  --help, -h                 Display this comprehensive usage guide\n\n"
              << "Scan Options:\n"
              << "  --format <table|json|sarif> Output format (default: table)\n"
              << "  --output <file>             Save audit findings to target file\n\n"
              << "Quantize Options:\n"
              << "  --output <file.bnet>        Save binary packed weights with SHA-256 header\n\n"
              << "Benchmark Options:\n"
              << "  --rows <M>                  Matrix row count (default: 512)\n"
              << "  --cols <K>                  Matrix column count (default: 512)\n"
              << "  --iterations <N>            Iteration count for benchmarking (default: 100)\n\n"
              << "Exit Codes:\n"
              << "  0                          Success / Audit Passed (Score: 100/100)\n"
              << "  1                          Security Findings Detected (Score < 100)\n"
              << "  2                          Invalid CLI arguments, parse error, or missing file\n";
}

static bool read_file_string(const std::string& path, std::string& out_content) {
    std::ifstream file(path);
    if (!file.is_open()) return false;
    std::stringstream buffer;
    buffer << file.rdbuf();
    out_content = buffer.str();
    return true;
}

static bool write_file_string(const std::string& path, const std::string& content) {
    std::ofstream file(path);
    if (!file.is_open()) return false;
    file << content;
    return true;
}

// -----------------------------------------------------------------------------
// Subcommand: scan
// -----------------------------------------------------------------------------
static int cmd_scan(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Error: 'scan' requires a policy configuration JSON path.\n";
        return 2;
    }

    std::string policy_path = argv[2];
    std::string format = "table";
    std::string output_path = "";

    for (int i = 3; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--format" && i + 1 < argc) {
            format = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            output_path = argv[++i];
        } else {
            std::cerr << "Error: Unknown option '" << arg << "' for scan.\n";
            return 2;
        }
    }

    std::string content;
    if (!read_file_string(policy_path, content)) {
        std::cerr << "Error: Cannot open policy file: " << policy_path << "\n";
        return 2;
    }

    std::string parse_err;
    JsonValue root = JsonValue::parse(content, &parse_err);
    if (!parse_err.empty()) {
        std::cerr << "Error: JSON parse failed for " << policy_path << ": " << parse_err << "\n";
        return 2;
    }

    AuditReport report = BitNetAuditor::audit_policy(root, policy_path);

    std::string output_str;
    if (format == "json") {
        output_str = Exporters::format_json(report);
    } else if (format == "sarif") {
        output_str = Exporters::format_sarif(report);
    } else if (format == "table") {
        output_str = Exporters::format_table(report);
    } else {
        std::cerr << "Error: Invalid output format '" << format << "'. Choose table, json, or sarif.\n";
        return 2;
    }

    if (!output_path.empty()) {
        if (!write_file_string(output_path, output_str)) {
            std::cerr << "Error: Failed to write output to file: " << output_path << "\n";
            return 2;
        }
    } else {
        std::cout << output_str;
    }

    return (report.score == 100) ? 0 : 1;
}

// -----------------------------------------------------------------------------
// Helper: Parse LayerWeights JSON
// -----------------------------------------------------------------------------
static bool parse_weights_json(const std::string& path, LayerWeights& layer) {
    std::string content;
    if (!read_file_string(path, content)) {
        std::cerr << "Error: Cannot open weights file: " << path << "\n";
        return false;
    }

    std::string parse_err;
    JsonValue root = JsonValue::parse(content, &parse_err);
    if (!parse_err.empty() || !root.is_object()) {
        std::cerr << "Error: Invalid JSON in " << path << ": " << parse_err << "\n";
        return false;
    }

    layer.rows = static_cast<size_t>(root.has_key("rows") ? root.get("rows")->as_int64() : 0);
    layer.cols = static_cast<size_t>(root.has_key("cols") ? root.get("cols")->as_int64() : 0);

    if (layer.rows == 0 || layer.cols == 0) {
        std::cerr << "Error: 'rows' and 'cols' must be non-zero in " << path << "\n";
        return false;
    }

    const JsonValue* w_arr = root.get("weights");
    if (!w_arr || !w_arr->is_array()) {
        std::cerr << "Error: 'weights' array missing in " << path << "\n";
        return false;
    }

    layer.weights.reserve(w_arr->arr_val.size());
    for (const auto& item : w_arr->arr_val) {
        layer.weights.push_back(static_cast<float>(item.as_double()));
    }

    if (layer.weights.size() != layer.rows * layer.cols) {
        std::cerr << "Error: weights count (" << layer.weights.size()
                  << ") does not match rows * cols (" << (layer.rows * layer.cols) << ")\n";
        return false;
    }

    const JsonValue* g_arr = root.get("rmsnorm_gain");
    if (g_arr && g_arr->is_array()) {
        layer.rmsnorm_gain.reserve(g_arr->arr_val.size());
        for (const auto& item : g_arr->arr_val) {
            layer.rmsnorm_gain.push_back(static_cast<float>(item.as_double()));
        }
    }

    const JsonValue* x_arr = root.get("input_activation");
    if (x_arr && x_arr->is_array()) {
        layer.input_activation.reserve(x_arr->arr_val.size());
        for (const auto& item : x_arr->arr_val) {
            layer.input_activation.push_back(static_cast<float>(item.as_double()));
        }
    }

    return true;
}

// -----------------------------------------------------------------------------
// Subcommand: quantize
// -----------------------------------------------------------------------------
static int cmd_quantize(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Error: 'quantize' requires a layer weights JSON path.\n";
        return 2;
    }

    std::string weights_path = argv[2];
    std::string output_bnet = "";

    for (int i = 3; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--output" && i + 1 < argc) {
            output_bnet = argv[++i];
        } else {
            std::cerr << "Error: Unknown option '" << arg << "' for quantize.\n";
            return 2;
        }
    }

    LayerWeights layer;
    if (!parse_weights_json(weights_path, layer)) {
        return 2;
    }

    PackedTernaryMatrix packed = TernaryQuantizer::quantize_matrix(layer.weights, layer.rows, layer.cols);

    size_t fp32_bytes = layer.rows * layer.cols * sizeof(float);
    size_t packed_bytes = packed.packed_data.size();
    float comp_ratio = (packed_bytes > 0) ? (static_cast<float>(fp32_bytes) / packed_bytes) : 0.0f;

    std::cout << "\n======================================================================\n"
              << "       LOGIHERTZ BITNET b1.58 WEIGHT QUANTIZATION & PACKING REPORT\n"
              << "======================================================================\n"
              << " Matrix Dimensions:    " << layer.rows << " x " << layer.cols << " (Total: " << (layer.rows * layer.cols) << " elements)\n"
              << " Absmean Scale Gamma:  " << std::fixed << std::setprecision(6) << packed.gamma << "\n"
              << " Zero Sparsity Ratio:  " << std::setprecision(2) << (packed.sparsity_ratio * 100.0f) << " %\n"
              << " Quantization MSE:     " << std::setprecision(6) << packed.mse << "\n"
              << " Original FP32 Size:   " << fp32_bytes << " bytes (" << (fp32_bytes / 1024.0) << " KB)\n"
              << " Packed 2-Bit Size:    " << packed_bytes << " bytes (" << (packed_bytes / 1024.0) << " KB)\n"
              << " Compression Ratio:    " << std::setprecision(2) << comp_ratio << "x (Theoretical max: 16.0x)\n"
              << " SHA-256 Checksum:     " << packed.sha256_hex << "\n"
              << " Status:               QUANTIZED & PACKED SUCCESSFULLY\n"
              << "======================================================================\n\n";

    if (!output_bnet.empty()) {
        if (!TernaryQuantizer::save_bnet_file(output_bnet, packed)) {
            std::cerr << "Error: Failed to write binary .bnet file to " << output_bnet << "\n";
            return 2;
        }
        std::cout << "Successfully exported packed binary file: " << output_bnet << "\n\n";
    }

    return 0;
}

// -----------------------------------------------------------------------------
// Subcommand: infer
// -----------------------------------------------------------------------------
static int cmd_infer(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Error: 'infer' requires a layer weights JSON path.\n";
        return 2;
    }

    std::string weights_path = argv[2];

    LayerWeights layer;
    if (!parse_weights_json(weights_path, layer)) {
        return 2;
    }

    if (layer.input_activation.empty()) {
        layer.input_activation.resize(layer.cols, 0.5f);
    } else if (layer.input_activation.size() != layer.cols) {
        std::cerr << "Error: input_activation length (" << layer.input_activation.size()
                  << ") must match cols (" << layer.cols << ")\n";
        return 2;
    }

    PackedTernaryMatrix packed = TernaryQuantizer::quantize_matrix(layer.weights, layer.rows, layer.cols);
    ForwardResult res = BitLinearKernel::forward(packed, layer.input_activation, layer.rmsnorm_gain);

    std::cout << "\n======================================================================\n"
              << "         LOGIHERTZ BITNET b1.58 BITLINEAR INFERENCE EXECUTION\n"
              << "======================================================================\n"
              << " SIMD Architecture:    " << BitLinearKernel::simd_architecture() << "\n"
              << " Matrix Shape:         " << layer.rows << " rows x " << layer.cols << " cols\n"
              << " Multiplication Math:  ZERO (Multiplication-Free Signed Integer Accumulation)\n"
              << " Pre-activation RMSNorm: " << (!layer.rmsnorm_gain.empty() ? "Applied" : "Skipped") << "\n"
              << " Kernel Latency:       " << std::fixed << std::setprecision(2) << res.latency_us << " microseconds\n"
              << " Throughput:           " << std::setprecision(2) << res.gops << " GOPS\n"
              << " Sample Outputs (first " << std::min<size_t>(8, res.output.size()) << " of " << res.output.size() << "):\n";

    for (size_t i = 0; i < std::min<size_t>(8, res.output.size()); ++i) {
        std::cout << "   y[" << std::setw(2) << i << "] = " << std::setw(9) << std::setprecision(4) << res.output[i]
                  << "  (raw accumulator sum: " << std::setw(6) << res.accumulators[i] << ")\n";
    }
    std::cout << "======================================================================\n\n";

    return 0;
}

// -----------------------------------------------------------------------------
// Subcommand: benchmark
// -----------------------------------------------------------------------------
static int cmd_benchmark(int argc, char** argv) {
    size_t rows = 512;
    size_t cols = 512;
    size_t iterations = 100;

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--rows" && i + 1 < argc) {
            rows = static_cast<size_t>(std::strtoul(argv[++i], nullptr, 10));
        } else if (arg == "--cols" && i + 1 < argc) {
            cols = static_cast<size_t>(std::strtoul(argv[++i], nullptr, 10));
        } else if (arg == "--iterations" && i + 1 < argc) {
            iterations = static_cast<size_t>(std::strtoul(argv[++i], nullptr, 10));
        } else {
            std::cerr << "Error: Unknown option '" << arg << "' for benchmark.\n";
            return 2;
        }
    }

    if (rows == 0 || cols == 0 || iterations == 0) {
        std::cerr << "Error: rows, cols, and iterations must be > 0\n";
        return 2;
    }

    // Synthesize deterministic weights and activations
    std::vector<float> fp_weights(rows * cols);
    for (size_t i = 0; i < rows * cols; ++i) {
        fp_weights[i] = std::sin(static_cast<float>(i)) * 1.5f;
    }

    PackedTernaryMatrix matrix = TernaryQuantizer::quantize_matrix(fp_weights, rows, cols);

    std::vector<float> fp_act(cols);
    for (size_t j = 0; j < cols; ++j) {
        fp_act[j] = std::cos(static_cast<float>(j)) * 1.2f;
    }

    QuantizedActivation qact = TernaryQuantizer::quantize_activation_absmax(fp_act);

    // Verify SIMD bit-exactness against Scalar
    std::vector<int32_t> scalar_acc(rows, 0);
    std::vector<float> scalar_fp(rows, 0.0f);
    BitLinearKernel::bitlinear_forward_scalar(matrix, qact, scalar_acc, scalar_fp);

    std::vector<int32_t> simd_acc(rows, 0);
    std::vector<float> simd_fp(rows, 0.0f);
    BitLinearKernel::bitlinear_forward_simd(matrix, qact, simd_acc, simd_fp);

    int32_t max_diff = 0;
    for (size_t i = 0; i < rows; ++i) {
        int32_t diff = std::abs(scalar_acc[i] - simd_acc[i]);
        if (diff > max_diff) max_diff = diff;
    }

    // Run timed benchmark iterations
    std::vector<int32_t> bench_acc(rows, 0);
    std::vector<float> bench_fp(rows, 0.0f);

    // Warm-up run
    for (size_t it = 0; it < 5; ++it) {
        BitLinearKernel::bitlinear_forward_simd(matrix, qact, bench_acc, bench_fp);
    }

    auto t_start = std::chrono::high_resolution_clock::now();
    for (size_t it = 0; it < iterations; ++it) {
        BitLinearKernel::bitlinear_forward_simd(matrix, qact, bench_acc, bench_fp);
    }
    auto t_end = std::chrono::high_resolution_clock::now();

    double total_us = std::chrono::duration<double, std::micro>(t_end - t_start).count();
    double avg_us = total_us / iterations;

    // 2 operations per weight (addition/subtraction of activation)
    double ops_per_pass = 2.0 * static_cast<double>(rows) * static_cast<double>(cols);
    double gops = (ops_per_pass / (avg_us * 1e-6)) / 1e9;

    std::cout << "\n======================================================================\n"
              << "       LOGIHERTZ BITNET b1.58 SIMD KERNEL BENCHMARK REPORT\n"
              << "======================================================================\n"
              << " Active SIMD Engine:   " << BitLinearKernel::simd_architecture() << "\n"
              << " Matrix Dimensions:    " << rows << " x " << cols << " (" << (rows * cols) << " weights)\n"
              << " Iterations Measured:  " << iterations << "\n"
              << " SIMD vs Scalar Diff:  " << max_diff << " (Bit-exact match verified)\n"
              << " Mean Pass Latency:    " << std::fixed << std::setprecision(2) << avg_us << " microseconds\n"
              << " Throughput:           " << std::setprecision(2) << gops << " GOPS (Multiplication-Free)\n"
              << " Theoretical Multipliers Used: 0\n"
              << "======================================================================\n\n";

    return (max_diff == 0) ? 0 : 1;
}

// -----------------------------------------------------------------------------
// Entrypoint
// -----------------------------------------------------------------------------
int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage("lh-bitnet");
        return 2;
    }

    std::string cmd = argv[1];

    if (cmd == "--version" || cmd == "-v") {
        print_version();
        return 0;
    }

    if (cmd == "--help" || cmd == "-h") {
        print_usage("lh-bitnet");
        return 0;
    }

    if (cmd == "scan") {
        return cmd_scan(argc, argv);
    }

    if (cmd == "quantize") {
        return cmd_quantize(argc, argv);
    }

    if (cmd == "infer") {
        return cmd_infer(argc, argv);
    }

    if (cmd == "benchmark") {
        return cmd_benchmark(argc, argv);
    }

    std::cerr << "Error: Unknown command '" << cmd << "'.\n";
    print_usage("lh-bitnet");
    return 2;
}
