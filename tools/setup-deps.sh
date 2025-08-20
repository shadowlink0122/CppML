#!/bin/bash
# MLLib Dependencies Setup Script
# Supports Linux, macOS, Windows (Git Bash/MSYS2/WSL)

set -e

echo "🚀 MLLib Dependencies Setup"
echo "=========================="

# Configuration
THIRD_PARTY_DIR="include/MLLib/third_party"
JSON_HPP="${THIRD_PARTY_DIR}/json.hpp"
JSON_URL="https://github.com/nlohmann/json/releases/download/v3.11.3/json.hpp"
JSON_ALT_URL="https://raw.githubusercontent.com/nlohmann/json/develop/single_include/nlohmann/json.hpp"

# Platform detection
if [[ "$OSTYPE" == "darwin"* ]]; then
    PLATFORM="macOS"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    PLATFORM="Linux"
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    PLATFORM="Windows"
else
    PLATFORM="Unknown"
fi

echo "Detected platform: ${PLATFORM}"

# Create directory
echo "📁 Creating third-party directory..."
mkdir -p "${THIRD_PARTY_DIR}"

# Check if already exists
if [[ -f "${JSON_HPP}" ]]; then
    echo "✅ ${JSON_HPP} already exists"
    exit 0
fi

# Download function
download_json() {
    local url=$1
    echo "📥 Downloading from: ${url}"
    
    if command -v curl >/dev/null 2>&1; then
        echo "Using curl..."
        curl -L -s --fail "${url}" -o "${JSON_HPP}"
        return $?
    elif command -v wget >/dev/null 2>&1; then
        echo "Using wget..."
        wget -q "${url}" -O "${JSON_HPP}"
        return $?
    else
        echo "❌ Neither curl nor wget is available"
        return 1
    fi
}

# Try primary URL, then alternative
echo "📥 Downloading nlohmann/json..."
if download_json "${JSON_URL}"; then
    echo "✅ Downloaded from primary source"
elif download_json "${JSON_ALT_URL}"; then
    echo "✅ Downloaded from alternative source"
else
    echo "❌ Download failed from both sources"
    echo ""
    echo "Manual installation required:"
    echo "1. Download from: ${JSON_URL}"
    echo "2. Save to: ${JSON_HPP}"
    echo ""
    echo "Platform-specific instructions:"
    case $PLATFORM in
        "Linux")
            echo "  sudo apt-get install curl wget  # Ubuntu/Debian"
            echo "  sudo yum install curl wget      # CentOS/RHEL"
            ;;
        "macOS")
            echo "  curl is built-in on macOS"
            echo "  brew install wget              # Optional"
            ;;
        "Windows")
            echo "  Use Git Bash or install curl for Windows"
            ;;
    esac
    exit 1
fi

# Verify download
if [[ -f "${JSON_HPP}" ]] && [[ -s "${JSON_HPP}" ]]; then
    echo "✅ JSON dependency successfully installed"
    echo "📊 File size: $(du -h "${JSON_HPP}" | cut -f1)"
    echo ""
    echo "Next steps:"
    echo "  make               # Build with JSON support"
    echo "  make test          # Run tests"
    echo "  make minimal       # Or build without JSON loading"
else
    echo "❌ Download verification failed"
    rm -f "${JSON_HPP}"
    exit 1
fi
