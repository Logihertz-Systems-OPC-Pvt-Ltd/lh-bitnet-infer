// ==============================================================================
// Product:       Logihertz BitNet 1.58-Bit Inference Engine (lh-bitnet-infer)
// Module:        include/bitnet_auditor.hpp
// Role:          10-Rule Deterministic Edge LLM Posture Auditor
// Organization:  Logihertz Systems (OPC) Private Limited
// Developer:     Sameer Tawade (Creator, Founder, Director & Sole Developer)
// Contact:       ops@logihertz.com / security@logihertz.com
// License:       Apache-2.0
// ==============================================================================

#ifndef LH_BITNET_AUDITOR_HPP
#define LH_BITNET_AUDITOR_HPP

#include "bitnet_types.hpp"
#include "json_parser.hpp"
#include <vector>
#include <string>

namespace logihertz {

class BitNetAuditor {
public:
    /// Get the official 10 normative rule registry
    static const std::vector<RuleDefinition>& get_rules();

    /// Evaluate deployment policy configuration against all 10 rules
    static AuditReport audit_policy(const JsonValue& policy, const std::string& manifest_path = "");
};

} // namespace logihertz

#endif // LH_BITNET_AUDITOR_HPP
