Name:           lh-bitnet-infer
Version:        0.1.0
Release:        1%{?dist}
Summary:        lh-bitnet - 1.58-Bit Ternary BitLinear Matrix Arithmetic
License:        Apache-2.0
URL:            https://github.com/Logihertz-Systems-OPC-Pvt-Ltd/lh-bitnet-infer
Source0:        %{name}-%{version}-src.tar.gz
Packager:       Sameer Tawade <ops@logihertz.com>
BuildArch:      %{_arch}

%description
C++17 1.58-bit ternary neural network (BitNet b1.58) inference engine featuring 2-bit weight packing (4 weights/byte), RMSNorm, multiplication-free SIMD (AVX2/NEON) BitLinear kernels, and Edge LLM auditor.

%prep
%autosetup -n %{name}-0.1.0

%build
# Industrial clean-room build with zero external runtime dependencies

%install
rm -rf %{buildroot}
mkdir -p %{buildroot}%{_bindir}
mkdir -p %{buildroot}%{_docdir}/%{name}
mkdir -p %{buildroot}%{_mandir}/man1

if [ -f bin/lh-bitnet ]; then
    install -m 0755 bin/lh-bitnet %{buildroot}%{_bindir}/lh-bitnet
elif [ -f lh-bitnet ]; then
    install -m 0755 lh-bitnet %{buildroot}%{_bindir}/lh-bitnet
fi

install -m 0644 README.md %{buildroot}%{_docdir}/%{name}/README.md
install -m 0644 SPEC.md %{buildroot}%{_docdir}/%{name}/SPEC.md
install -m 0644 LICENSE %{buildroot}%{_docdir}/%{name}/LICENSE
install -m 0644 NOTICE %{buildroot}%{_docdir}/%{name}/NOTICE

%files
%doc %{_docdir}/%{name}/README.md
%doc %{_docdir}/%{name}/SPEC.md
%doc %{_docdir}/%{name}/NOTICE
%license %{_docdir}/%{name}/LICENSE

%changelog
* Mon Nov 16 2026 Sameer Tawade <ops@logihertz.com> - 0.1.0-1
- Initial production release for Logihertz Systems Ecosystem and STRUCT OS.
