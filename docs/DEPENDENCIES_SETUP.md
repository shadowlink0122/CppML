# MLLib Dependencies Setup Guide

## 🚀 Quick Start (Automatic)

MLLib automatically manages its dependencies. Just run:

```bash
git clone https://github.com/shadowlink0122/CppML.git
cd CppML
make setup-deps    # Automatically downloads required libraries
make               # Build with full JSON support
```

## 📋 Build Options

### Option 1: Full Features (Recommended)
```bash
make               # Automatic dependency download + full build
make test          # Run all tests including JSON I/O
```

### Option 2: Minimal Build (No Dependencies)
```bash
make minimal       # Build without JSON loading support
                   # JSON saving still works (no dependencies required)
```

### Option 3: Manual Dependency Management
```bash
make deps-check    # Check what dependencies are missing
make deps-install  # Install missing dependencies
make build         # Build with available features
```

## 🌍 Platform-Specific Instructions

### Linux (Ubuntu/Debian)
```bash
# Automatic (recommended)
make

# Manual installation
sudo apt-get update
sudo apt-get install curl wget
make setup-deps
```

### Linux (CentOS/RHEL/Fedora)
```bash
# Automatic (recommended)
make

# Manual installation
sudo yum install curl wget  # CentOS/RHEL
# or
sudo dnf install curl wget  # Fedora
make setup-deps
```

### macOS
```bash
# Automatic (recommended)
make

# Manual with Homebrew (optional)
brew install curl wget
make setup-deps

# Manual with curl (built-in)
make setup-deps  # Uses built-in curl
```

### Windows
```bash
# Git Bash / MSYS2 / WSL
make

# Manual download (if automatic fails)
curl -L https://github.com/nlohmann/json/releases/download/v3.11.3/json.hpp -o include/MLLib/third_party/json.hpp
make build
```

## 🔧 Troubleshooting

### Network Issues
```bash
# If automatic download fails:
mkdir -p include/MLLib/third_party
wget https://github.com/nlohmann/json/releases/download/v3.11.3/json.hpp -O include/MLLib/third_party/json.hpp

# Alternative mirrors
curl -L https://raw.githubusercontent.com/nlohmann/json/develop/single_include/nlohmann/json.hpp -o include/MLLib/third_party/json.hpp
```

### Offline Environment
```bash
# 1. Download on internet-connected machine:
curl -L https://github.com/nlohmann/json/releases/download/v3.11.3/json.hpp -o json.hpp

# 2. Copy to offline machine:
mkdir -p include/MLLib/third_party
cp json.hpp include/MLLib/third_party/

# 3. Build offline:
make build
```

### Corporate Firewall
```bash
# Use corporate proxy
export https_proxy=your-proxy:port
make setup-deps

# Or download manually and place in include/MLLib/third_party/
```

## 📊 Feature Matrix by Build Type

| Build Type | Binary I/O | JSON Save | JSON Load | Dependencies |
|------------|------------|-----------|-----------|--------------|
| **Full** | ✅ | ✅ | ✅ | Auto-managed |
| **Minimal** | ✅ | ✅ | ❌ | None |
| **Custom** | ✅ | ✅ | ⚠️ | User choice |

## 🧪 Verification

Test your setup:
```bash
# Quick test
make test-deps     # Verify dependencies
make quick-test    # Basic functionality test
make full-test     # Complete test suite

# Feature-specific tests
make test-json     # JSON I/O tests
make test-binary   # Binary I/O tests
```

## ⚠️ Known Issues

### Issue: curl/wget not available
**Solution:**
```bash
# Install curl or wget first
# Ubuntu/Debian: sudo apt-get install curl
# CentOS/RHEL: sudo yum install curl
# macOS: curl is built-in
# Windows: Use Git Bash or install curl
```

### Issue: Permission denied
**Solution:**
```bash
# Make sure you have write permission
sudo chown -R $USER:$USER include/
make setup-deps
```

### Issue: Corporate network blocks GitHub
**Solution:**
```bash
# Download manually from alternative source
curl -L https://raw.githubusercontent.com/nlohmann/json/v3.11.3/single_include/nlohmann/json.hpp -o include/MLLib/third_party/json.hpp
```

## 🤝 Support

If you encounter issues:

1. **Check build logs**: `make verbose`
2. **Verify dependencies**: `make deps-check`
3. **Try minimal build**: `make minimal`
4. **Report issues**: [GitHub Issues](https://github.com/shadowlink0122/CppML/issues)

## 📄 Dependency Licenses

- **nlohmann/json**: MIT License
- Compatible with MLLib's BSD 3-Clause with Commercial Use Restriction
