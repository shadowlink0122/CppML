# MLLib - C++ Machine Learning Library
> **Language**: 🇺🇸 English | [🇯🇵 日本語](README_ja.md)

[![CI](https://github.com/shadowlink0122/CppML/workflows/CI/badge.svg)](https://github.com/shadowlink0122/CppML/actions/workflows/ci.yml)
[![Extended CI](https://github.com/shadowlink0122/CppML/workflows/Extended%20CI/badge.svg)](https://github.com/shadowlink0122/CppML/actions/workflows/extended-ci.yml)
[![GPU CI](https://github.com/shadowlink0122/CppML/workflows/GPU%20CI/badge.svg)](https://github.com/shadowlink0122/CppML/actions/workflows/gpu-ci.yml)
[![Code Quality](https://img.shields.io/badge/code%20style-K%26R-blue.svg)](https://en.wikipedia.org/wiki/Indentation_style#K&R_style)
[![Tests](https://img.shields.io/badge/tests-21%2F21_unit_tests-brightgreen.svg)](#-testing)
[![Integration Tests](https://img.shields.io/badge/integration-3429%2F3429_assertions-brightgreen.svg)](#-testing)
[![GPU Tests](https://img.shields.io/badge/GPU_tests-145_assertions-blue.svg)](#-gpu-support)
[![Test Coverage](https://img.shields.io/badge/coverage-100%25_CI_success-brightgreen.svg)](#-testing)

> **Languages**: [English](README.md) | [日本語](README_ja.md)

A modern C++17 machine learning library designed for neural network training and model persistence with comprehensive testing and performance monitoring.

## ✨ Features

- **🧠 Neural Networks**: Sequential models with customizable architectures
- **📊 Layers**: Dense (fully connected), ReLU, Sigmoid, Tanh activation functions  
- **🎯 Training**: MSE loss function with SGD optimizer
- **🎯 Multi-GPU Support**: NVIDIA CUDA, AMD ROCm, Intel oneAPI, Apple Metal with automatic detection
- **�💾 Model I/O**: Save/load models in binary, JSON, and config formats
- **📁 Auto Directory**: Automatic directory creation with `mkdir -p` functionality
- **🔧 Type Safety**: Enum-based format specification for improved reliability
- **⚡ Performance**: Optimized C++17 implementation with NDArray backend
- **🧪 Testing**: Comprehensive unit (21/21) and integration tests (3429/3429 assertions)
- **🖥️ GPU Testing**: 145 GPU-specific assertions with fallback validation
- **🔄 Cross-platform**: Linux, macOS, Windows support
- **📊 Benchmarking**: Real-time performance metrics and execution time tracking
- **🎯 CI/CD Ready**: 100% test success rate for production deployments

## 🚀 Quick Start

### Prerequisites

- C++17 compatible compiler (GCC 7+, Clang 5+, MSVC 2017+)
- Make

### Build and Test

```bash
git clone https://github.com/shadowlink0122/CppML.git
cd CppML
make                         # Build the library
make test                    # Run all tests (unit + integration)
make unit-test               # Run unit tests (21/21 passing)
make integration-test        # Run integration tests (3429/3429 assertions)
make simple-integration-test # Run simple integration tests

# CI-optimized testing (for faster execution)
make build-tests             # Build test executables only (for CI artifacts)
make unit-test-run-only      # Run unit tests with pre-built executables
make integration-test-run-only # Run integration tests with pre-built executables

make samples                 # Build all sample programs
make xor                     # Run XOR neural network example
make device-detection        # Run GPU device detection sample
make gpu-vendor-detection    # Run GPU vendor detection sample
```

### Basic Usage

```cpp
#include "MLLib.hpp"
using namespace MLLib;

// Create XOR neural network
model::Sequential model;

// Add layers
model.add(std::make_shared<layer::Dense>(2, 4));
model.add(std::make_shared<layer::activation::ReLU>());
model.add(std::make_shared<layer::Dense>(4, 1));
model.add(std::make_shared<layer::activation::Sigmoid>());

// Training data
std::vector<std::vector<double>> X = {{0,0}, {0,1}, {1,0}, {1,1}};
std::vector<std::vector<double>> Y = {{0}, {1}, {1}, {0}};

// Train model
loss::MSE loss;
optimizer::SGD optimizer(0.1);
model.train(X, Y, loss, optimizer, [](int epoch, double loss) {
    printf("Epoch %d, Loss: %f\n", epoch, loss);
}, 1000);

// Make predictions (multiple syntax options supported)
auto pred1 = model.predict(std::vector<double>{1.0, 0.0});  // Vector syntax
auto pred2 = model.predict({1.0, 0.0});                     // Initializer list syntax (C++17)

// Save model
model::ModelIO::save_model(model, "model.bin", model::ModelFormat::BINARY);
```

## 💾 Model I/O

MLLib supports multiple model serialization formats:

### Save Models

```cpp
using namespace MLLib;

// Enum-based format specification (type-safe)
model::ModelIO::save_model(model, "model.bin", model::ModelFormat::BINARY);
model::ModelIO::save_model(model, "model.json", model::ModelFormat::JSON);
model::ModelIO::save_model(model, "model.config", model::ModelFormat::CONFIG);

// String-based (legacy support)
auto format = model::ModelIO::string_to_format("binary");
model::ModelIO::save_model(model, "model.bin", format);
```

### Load Models

```cpp
// Load complete model with parameters
auto model = model::ModelIO::load_model("model.bin", model::ModelFormat::BINARY);

// Load only configuration (no trained parameters)
auto empty_model = model::ModelIO::load_config("model.config");
```

### Automatic Directory Creation

```cpp
// Automatically creates nested directories (like mkdir -p)
model::ModelIO::save_model(model, "models/experiment_1/epoch_100.bin", 
                          model::ModelFormat::BINARY);
```

## 🧪 Testing

MLLib includes a comprehensive testing framework with performance monitoring:

### Test Coverage

- **Unit Tests**: 21/21 tests passing across all components
- **Integration Tests**: End-to-end workflows and complex scenarios
- **Performance Tests**: Execution time monitoring and benchmarking
- **Error Handling**: Comprehensive error condition testing

### Test Categories

```bash
# Unit tests with execution time monitoring
make unit-test

# Sample output:
# Running test: ConfigConstantsTest
# ✅ ConfigConstantsTest PASSED (10 assertions, 0.03ms)
# Running test: NDArrayMatmulTest  
# ✅ NDArrayMatmulTest PASSED (11 assertions, 0.02ms)
# 
# Total test execution time: 0.45ms
# Total suite time (including overhead): 0.89ms
```

### Test Components

- **Config Module (3/3)**: Library configuration and constants
- **NDArray Module (6/6)**: Multi-dimensional array operations
- **Dense Layer (4/4)**: Fully connected layer functionality
- **Activation Functions (7/7)**: ReLU, Sigmoid, Tanh with error handling
- **Sequential Model (1/1)**: Model architecture and prediction

### Integration Tests

```bash
make integration-test          # Comprehensive integration tests (3429/3429 assertions)
make simple-integration-test   # Simple integration tests (basic functionality)

# Comprehensive Integration Test Coverage:
# 🎯 XOR Model Tests: Basic functionality + learning convergence (CI-safe)
# 🔧 Optimizer Integration: SGD + Adam fallback testing
# 📊 Loss Function Integration: MSE + CrossEntropy validation
# 💻 Backend Integration: CPU backend comprehensive testing (601 assertions)
# 🔗 Layer Integration: Dense layers + activation functions 
# 🛠️ Utility Integration: Matrix, Random, Validation utilities (504 assertions)
# 📱 Device Integration: CPU device operations (2039 assertions)
# 📁 Data Integration: Loading, batching, validation (157 assertions)
# ⚡ Performance Testing: Stability + execution time monitoring

# Test Results Summary:
# ✅ 100% CI success rate (3429/3429 assertions passing)
# ✅ All tests deterministic and CI-ready
# ✅ Comprehensive component integration coverage
```

## 🛠️ Development

### Testing

```bash
# Run all tests
make test                     # Complete test suite (Unit + Integration)
make unit-test                # Run unit tests (21/21 passing)
make integration-test         # Comprehensive integration tests (3429/3429 assertions)
make simple-integration-test  # Simple integration tests (basic functionality)

# Test output includes execution time monitoring:
# ✅ BasicXORModelTest PASSED (5 assertions, 0.17ms)
# ✅ AdamOptimizerIntegrationTest PASSED (10 assertions, 1.04ms)
# ✅ BackendPerformanceIntegrationTest PASSED (551 assertions, 43.54ms)
# 🎉 ALL INTEGRATION TESTS PASSED! (3429/3429 assertions, 100% success rate)
```

### Code Quality

```bash
# Format code
make fmt

# Check formatting
make fmt-check

# Run linting
make lint

# Static analysis
make check

# All quality checks
make lint-all
```

### Building

```bash
# Build library
make

# Debug build
make debug

# Build and run samples
make samples
make run-sample SAMPLE=xor           # Run specific sample using generic runner
make xor                            # XOR neural network (alias)
make device-detection               # GPU device detection (alias)
make gpu-vendor-detection           # GPU vendor detection (alias)

# Clean (removes training outputs)
make clean
```

### Install Development Tools

```bash
make install-tools
```

## 🎯 GPU Support

MLLib provides comprehensive multi-GPU vendor support with automatic detection and fallback:

### Features

- **🌍 Multi-Vendor**: NVIDIA CUDA, AMD ROCm, Intel oneAPI, Apple Metal
- **🔄 Auto Detection**: Runtime GPU detection and selection
- **💪 Default Support**: All GPU vendors enabled by default (library design)
- **�️ CPU Fallback**: Seamless CPU execution when GPU unavailable
- **⚠️ Smart Warnings**: Informative messages about GPU status
- **🧪 Full Testing**: 145 GPU assertions across unit and integration tests

### Supported GPU Vendors

| Vendor | API | Hardware Support |
|--------|-----|------------------|
| **NVIDIA** | CUDA, cuBLAS | GeForce, Quadro, Tesla, RTX |
| **AMD** | ROCm, HIP, hipBLAS | Radeon Instinct, Radeon Pro |
| **Intel** | oneAPI, SYCL, oneMKL | Arc, Iris Xe, UHD Graphics |
| **Apple** | Metal, MPS | M1, M1 Pro/Max/Ultra, M2 |

### GPU Build Options

```bash
# Default build (all GPU vendors enabled)
make

# Disable specific GPU support
make DISABLE_CUDA=1           # Disable NVIDIA CUDA
make DISABLE_ROCM=1           # Disable AMD ROCm
make DISABLE_ONEAPI=1         # Disable Intel oneAPI
make DISABLE_METAL=1          # Disable Apple Metal

# CPU-only build
make DISABLE_CUDA=1 DISABLE_ROCM=1 DISABLE_ONEAPI=1 DISABLE_METAL=1

# GPU testing with environment control
FORCE_CPU_ONLY=1 make test    # Force CPU-only testing
GPU_SIMULATION=1 make test    # Enable GPU simulation mode
```

### Usage

```cpp
#include "MLLib.hpp"

int main() {
    MLLib::model::Sequential model;
    
    // Set GPU device (automatic vendor detection)
    model.set_device(MLLib::DeviceType::GPU);
    // Library outputs: ✅ GPU device successfully configured
    // Or warnings: ⚠️ WARNING: GPU device requested but no GPU found!
    
    // Build neural network
    model.add_layer(new MLLib::layer::Dense(784, 128));
    model.add_layer(new MLLib::layer::activation::ReLU());
    model.add_layer(new MLLib::layer::Dense(128, 10));
    
    // Training automatically uses optimal GPU
    model.train(train_X, train_Y, loss, optimizer);
    
    return 0;
}
```

### GPU Status Check

```cpp
// Check GPU availability
if (MLLib::Device::isGPUAvailable()) {
    // Display detected GPUs
    auto gpus = MLLib::Device::detectGPUs();
    for (const auto& gpu : gpus) {
        printf("GPU: %s (%s)\n", gpu.name.c_str(), gpu.api_support.c_str());
    }
}
```

### GPU Documentation

- **📖 [Multi-GPU Support Guide (English)](docs/MULTI_GPU_SUPPORT_en.md)**
- **📖 [マルチGPUサポートガイド (日本語)](docs/MULTI_GPU_SUPPORT_ja.md)**
- **⚙️ [GPU CI Setup Guide](docs/GPU_CI_SETUP_en.md)**

## 📚 Documentation

Comprehensive documentation is available in the [`docs/`](docs/) directory:

- **📖 [Overview](docs/README.md)** - Documentation index and quick start
- **🇺🇸 [English Documentation](docs/README_en.md)** - Complete guide in English
- **🇯🇵 [日本語ドキュメント](docs/README_ja.md)** - 日本語での完全ガイド
- **🇺🇸 [Model I/O Guide (English)](docs/MODEL_IO_en.md)** - Complete model serialization guide
- **🇯🇵 [モデル I/O ガイド (日本語)](docs/MODEL_IO_ja.md)** - 日本語でのモデル保存・読み込み解説
- **🇺🇸 [GPU CI Setup Guide (English)](docs/GPU_CI_SETUP_en.md)** - GPU testing environment configuration
- **🇯🇵 [GPU CI 設定ガイド (日本語)](docs/GPU_CI_SETUP_ja.md)** - GPU テスト環境の設定方法
- **🇺🇸 [Testing Guide (English)](docs/TESTING_en.md)** - Testing framework documentation
- **🇯🇵 [テストガイド (日本語)](docs/TESTING_ja.md)** - テストフレームワークの説明

### API Components

- **Core**: `NDArray` (tensor operations), `DeviceType` (CPU/GPU management)
- **Layers**: `Dense` (fully connected), `ReLU`, `Sigmoid` activation functions
- **Models**: `Sequential` (layer stacking), `ModelIO` (serialization)
- **Training**: `MSELoss` (mean squared error), `SGD` (stochastic gradient descent)
- **Utils**: Automatic directory creation, cross-platform filesystem operations

### Architecture

```
MLLib/
├── NDArray          # Multi-dimensional arrays (tensor operations)
├── Device           # CPU/GPU device management
├── Layer/           # Neural network layers
│   ├── Dense        # Fully connected layers
│   └── Activation/  # Activation functions (ReLU, Sigmoid)
├── Loss/            # Loss functions (MSE)
├── Optimizer/       # Optimization algorithms (SGD)
└── Model/           # Model definition and I/O
    ├── Sequential   # Sequential model architecture
    └── ModelIO      # Model serialization (Binary/JSON/Config)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run quality checks (`make lint-all`)
5. Test your changes (`make samples`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

### Code Style

This project uses K&R style formatting. Please run `make fmt` before submitting.

## 🎯 Examples

### XOR Neural Network

The included XOR example demonstrates:
- Model architecture definition
- Training with callback functions
- Epoch-based model saving
- Multiple file format support

```bash
make xor  # Run the XOR neural network example
```

### Model Format Testing

Test all model I/O formats and sample programs:

```bash
make samples               # Build all sample programs
make run-sample            # List available samples
make run-sample SAMPLE=xor # Run specific sample
make xor                   # Run XOR neural network example (alias)
```

## � CI/CD & Quality Assurance

MLLib features a comprehensive CI/CD pipeline with 100% test success rate:

### GitHub Actions Pipeline

```yaml
# Complete CI pipeline workflow:
Format Validation → Linting → Static Analysis
         ↓
    Build Test → Unit Tests → Integration Tests
         ↓
     CI Summary
```

### Test Coverage

- **Unit Tests**: 21/21 passing (100%)
- **Integration Tests**: 3429/3429 assertions passing (100%)
- **Simple Integration Tests**: Basic functionality verification
- **CI Requirements**: 100% deterministic test success rate

### Quality Checks

```bash
make fmt-check    # Code formatting validation
make lint         # Clang-tidy linting 
make check        # Cppcheck static analysis
make lint-all     # All quality checks combined
```

### CI Features

- ✅ **Deterministic Testing**: All tests designed for CI reliability
- ✅ **Comprehensive Coverage**: End-to-end component integration testing
- ✅ **Performance Monitoring**: Execution time tracking for all tests
- ✅ **Multi-platform Support**: Ubuntu, macOS, Windows compatibility
- ✅ **Production Ready**: 100% test success rate for deployment confidence

## �📁 Project Structure

```
CppML/
├── include/MLLib/         # Header files
│   ├── config.hpp         # Library configuration
│   ├── ndarray.hpp        # Tensor operations
│   ├── device/            # Device management
│   ├── layer/             # Neural network layers
│   ├── loss/              # Loss functions
│   ├── optimizer/         # Optimization algorithms
│   └── model/             # Model architecture & I/O
├── src/MLLib/            # Implementation files
├── sample/               # Example programs
├── docs/                 # Documentation
└── README.md             # This file
```

## ⚠️ Current Limitations

- **GPU Support**: Currently CPU-only (GPU planned for future releases)
- **Layer Types**: Limited to Dense and basic activation functions
- **Optimizers**: Only SGD implemented (Adam, RMSprop planned)
- **JSON Loading**: JSON format supports saving but loading is not yet implemented

## 🛠️ Roadmap

### Completed ✅
- [x] Comprehensive unit testing framework (21/21 tests)
- [x] Integration testing with performance monitoring
- [x] Execution time measurement and benchmarking
- [x] Dense layers with activation functions (ReLU, Sigmoid, Tanh)
- [x] Sequential model architecture
- [x] MSE loss and SGD optimizer
- [x] Model save/load (binary, config formats)

### In Progress 🚧
- [ ] GPU acceleration support
- [ ] JSON model loading implementation
- [ ] Advanced error handling and validation

### Planned 📋
- [ ] Additional layer types (Convolutional, LSTM, Dropout)
- [ ] More optimizers (Adam, RMSprop, AdaGrad)
- [ ] Advanced loss functions (CrossEntropy, Huber)
- [ ] Model quantization and optimization
- [ ] Python bindings
- [ ] Multi-threading support

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by modern ML frameworks like PyTorch and TensorFlow
- Built with modern C++ best practices
- Designed for educational and research purposes
