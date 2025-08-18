# MLLib Documentation

> **Language**: 🇺🇸 English | [🇯🇵 日本語](README_ja.md)

[![CI](https://github.com/shadowlink0122/CppML/workflows/CI/badge.svg)](https://github.com/shadowlink0122/CppML/actions/workflows/ci.yml)
[![Extended CI](https://github.com/shadowlink0122/CppML/workflows/Extended%20CI/badge.svg)](https://github.com/shadowlink0122/CppML/actions/workflows/extended-ci.yml)
[![Code Quality](https://img.shields.io/badge/code%20style-K%26R-blue.svg)](https://en.wikipedia.org/wiki/Indentation_style#K&R_style)
[![Tests](https://img.shields.io/badge/tests-21%2F21_passing-brightgreen.svg)](#testing)
[![GPU Optimization](https://img.shields.io/badge/GPU_kernel_reduction-97%25-brightgreen.svg)](#gpu-strategy)
[![Model I/O Generalization](https://img.shields.io/badge/Model_I%2FO_generalization-92%25-green.svg)](#model-io-guides)

Welcome to the comprehensive documentation collection for MLLib C++ machine learning library.

## 📚 Main Documentation

### 🌐 Language-Specific Complete Guides

| Language | File | Description |
|----------|------|-------------|
| 🇺🇸 **English** | [GUIDE_en.md](GUIDE_en.md) | Complete guide in English |
| 🇯🇵 **Japanese** | [GUIDE_ja.md](GUIDE_ja.md) | 日本語での完全ガイド |
| 🏠 **Project** | [../README.md](../README.md) | Main project README |

## 🔧 Technical Documentation

### 💾 Model I/O Guides

| Language | File | Description |
|----------|------|-------------|
| 🇺🇸 **English** | [MODEL_IO_en.md](MODEL_IO_en.md) | Complete model serialization guide |
| 🇯🇵 **Japanese** | [MODEL_IO_ja.md](MODEL_IO_ja.md) | モデル保存・読み込みの完全ガイド |

### 🧪 Testing Guides

| Language | File | Description |
|----------|------|-------------|
| 🇺🇸 **English** | [TESTING_en.md](TESTING_en.md) | Testing framework documentation |
| 🇯🇵 **Japanese** | [TESTING_ja.md](TESTING_ja.md) | テストフレームワークの説明 |

### 🚀 GPU Documentation

| Language | File | Description |
|----------|------|-------------|
| 🇺🇸 **English** | [GPU_STRATEGY_en.md](GPU_STRATEGY_en.md) | GPU backend strategy and roadmap |
| 🇯🇵 **Japanese** | [GPU_STRATEGY_ja.md](GPU_STRATEGY_ja.md) | GPU バックエンド戦略とロードマップ |
| 🇺🇸 **English** | [MULTI_GPU_SUPPORT_en.md](MULTI_GPU_SUPPORT_en.md) | Multi-GPU vendor support |
| 🇯🇵 **Japanese** | [MULTI_GPU_SUPPORT_ja.md](MULTI_GPU_SUPPORT_ja.md) | マルチGPUベンダーサポート |
| 🇺🇸 **English** | [GPU_CI_SETUP_en.md](GPU_CI_SETUP_en.md) | GPU testing environment setup |
| 🇯🇵 **Japanese** | [GPU_CI_SETUP_ja.md](GPU_CI_SETUP_ja.md) | GPU テスト環境セットアップ |
| 🇺🇸 **English** | [GPU_DETECTION_GUIDE_en.md](GPU_DETECTION_GUIDE_en.md) | GPU detection verification guide |
| 🇯🇵 **Japanese** | [GPU_DETECTION_GUIDE_ja.md](GPU_DETECTION_GUIDE_ja.md) | GPU検出機能の検証ガイド |

## 🚀 Quick Access

### 📖 Getting Started

- **🏃‍♂️ [Quick Start (English)](GUIDE_en.md#-quick-start)** - Get started with MLLib in 5 minutes
- **🏃‍♀️ [クイックスタート (日本語)](GUIDE_ja.md#-クイックスタート)** - 5分で始めるMLLib
- **🔧 [Build Instructions (English)](GUIDE_en.md#build-and-test)** - How to build the project
- **🎯 [Basic Usage Examples](GUIDE_en.md#basic-usage)** - XOR network implementation

### 🧪 Testing Resources

- **📊 [How to Run Tests](TESTING_en.md)** - Unit and integration test execution
- **⏱️ [Performance Monitoring](TESTING_en.md#performance-monitoring)** - Runtime measurement features
- **✅ [Test Coverage](TESTING_en.md#test-coverage)** - 21/21 test passing status

### 💾 Model Operations

- **💿 [Model Saving](MODEL_IO_en.md#model-saving)** - Binary, JSON, Config formats
- **📥 [Model Loading](MODEL_IO_en.md#model-loading)** - Using pre-trained models
- **🔒 [Type Safety](MODEL_IO_en.md#type-safety)** - Enum-based safe operations

### 🚀 GPU Support

- **⚡ [GPU Strategy](GPU_STRATEGY_en.md)** - Complete coverage plan for all GPU vendors
- **🔧 [Multi-GPU Support](MULTI_GPU_SUPPORT_en.md)** - NVIDIA, AMD, Intel, Apple GPU support
- **🧪 [GPU CI Setup](GPU_CI_SETUP_en.md)** - Automated GPU testing configuration
- **🔍 [GPU Detection Guide](GPU_DETECTION_GUIDE_en.md)** - GPU detection verification methods

## 🏗️ Architecture

```
MLLib/
├── NDArray          # Multi-dimensional arrays (tensor operations)
├── Device           # CPU/GPU device management
├── Layer/           # Neural network layers
│   ├── Dense        # Fully connected layers
│   └── Activation/  # Activation functions (ReLU, Sigmoid, Tanh)
├── Loss/            # Loss functions (MSE)
├── Optimizer/       # Optimization algorithms (SGD)
└── Model/           # Model definition and I/O
    ├── Sequential   # Sequential model architecture
    └── ModelIO      # Model serialization
```

## 📁 Documentation Structure

```
docs/
├── README.md           # This file (documentation index)
├── README_en.md        # English complete guide
├── README_ja.md        # Japanese complete guide
├── MODEL_IO_en.md      # Model I/O guide (English)
├── MODEL_IO_ja.md      # Model I/O guide (Japanese)
├── TESTING_en.md       # Testing guide (English)
├── TESTING_ja.md       # Testing guide (Japanese)
├── GPU_STRATEGY_en.md  # GPU strategy (English)
├── GPU_STRATEGY_ja.md  # GPU strategy (Japanese)
└── MULTI_GPU_*         # Multi-GPU documentation
```

## 🎯 Examples and Samples

### 💡 Basic Example

```cpp
// Simple XOR neural network example
#include "MLLib.hpp"
using namespace MLLib;

model::Sequential model;
model.add(std::make_shared<layer::Dense>(2, 4));
model.add(std::make_shared<layer::activation::ReLU>());
model.add(std::make_shared<layer::Dense>(4, 1));
model.add(std::make_shared<layer::activation::Sigmoid>());

// See README_en.md for details
```

### 🔧 Executable Samples

```bash
# Run from project root
make samples               # Build all sample programs
make run-sample            # Display available samples list
make xor                   # XOR network example (alias)
make device-detection      # GPU device detection sample (alias)
make gpu-usage-test        # GPU performance test sample (alias)
make unit-test             # Run all unit tests
```

## 🛠️ Developer Information

### 📋 Requirements

- **Compiler**: C++17 compatible (GCC 7+, Clang 5+, MSVC 2017+)
- **Build System**: Make
- **Platforms**: Linux, macOS, Windows

### 🔍 Quality Assurance

- **🧪 Tests**: 21/21 tests passing (with runtime monitoring)
- **📊 Coverage**: Comprehensive unit and integration tests
- **🔧 Static Analysis**: cppcheck, clang-tidy support
- **📝 Code Style**: K&R style formatting

## 🤝 Contributing

1. **📖 [Contribution Guide](README_en.md#-contributing)** - How to participate
2. **🎨 [Code Style](README_en.md#code-style)** - K&R formatting guidelines
3. **✅ [Quality Checks](README_en.md#code-quality)** - Linting and testing requirements

## 📧 Support

- **🐛 [Issues](https://github.com/shadowlink0122/CppML/issues)** - Bug reports and feature requests
- **💬 [Discussions](https://github.com/shadowlink0122/CppML/discussions)** - Questions and discussions
- **📋 [Projects](https://github.com/shadowlink0122/CppML/projects)** - Development roadmap

## 📄 License

This project is released under the [MIT License](../LICENSE).

---

**💡 Tip**: First-time users are recommended to start with the [English Complete Guide](GUIDE_en.md) or [日本語完全ガイド](GUIDE_ja.md).
