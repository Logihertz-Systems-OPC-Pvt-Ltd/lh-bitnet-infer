# Security Policy — Logihertz BitNet Inference Engine

## Supported Versions

Logihertz Systems (OPC) Private Limited maintains and publishes security updates for the following active releases of `lh-bitnet-infer`:

| Version | Supported          | Security Maintenance Tier |
|:-------:|:------------------:|:--------------------------|
| 0.1.x   | :white_check_mark: | Active Production Release |

## Threat Model & Scope

`lh-bitnet-infer` operates as an edge-native Large Language Model (LLM) and neural inference runtime for sovereign embedded appliances, robotics, and edge servers (such as NVIDIA Jetson Orin and industrial ARM64/x86_64 nodes). Security considerations include:

1. **Weight Integrity & Trojan Mitigation**: Preventing malicious weight corruption or backdoor injections by verifying FIPS 180-4 SHA-256 cryptographic digests before deserialization into memory.
2. **Buffer Bounds & Integer Safety**: Enforcing strict bounds checking across 2-bit packed weight stream unpacking and preventing signed 32-bit integer overflow during ternary dot-product accumulations.
3. **KV-Cache Memory Bounding & Zeroization**: Mitigating Denial of Service (OOM) via strict context token ceilings and preventing cross-session prompt telemetry leakage through mandatory `secure_zero_memory()` wipe routines on session termination.
4. **Local Air-Gap RPC Security**: Requiring mutual TLS and token authorization whenever edge RPC servers bind to non-loopback network interfaces.
5. **Memory Locking**: Supporting `mlock()` to prevent sensitive model weights and KV-cache states from being paged to unencrypted swap storage.

## Reporting a Vulnerability

If you discover a potential vulnerability or security flaw in `lh-bitnet-infer`, please submit a confidential disclosure:

- **Primary Security Channel**: `security@logihertz.com`
- **Operations & Founder Escalation**: `ops@logihertz.com` (Direct attention: Sameer Tawade)
- **Response SLA**: Initial triage and acknowledgment within 24–48 hours.
- **Remediation SLA**: Coordinated patch development, verification, and tagged release within 14 business days.

Please do not open public GitHub issues for undisclosed security vulnerabilities. All disclosures will be credited in release notes upon coordinated remediation.
