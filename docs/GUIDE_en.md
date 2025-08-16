# MLLib - C++ Machine Learning Library Complete Guide

> **Language**: 🇺🇸 English | [🇯🇵 日本語](GUIDE_ja.md)

[![CI](https://github.com/shadowlink0122/CppML/workf# ✅ GPU is 25.0x faster than CPU! (Excellent acceleration)
```

## 🧪 Testing

MLLib includes a comprehensive testing framework with performance monitoring:

### Test Coverage

- **Unit Tests**: 21/21 tests passing across all components
- **Integration Tests**: End-to-end workflows and complex scenarios  
- **Performance Tests**: Execution time monitoring and benchmarking
- **Error Handling**: Comprehensive error condition testing
- **GPU Tests**: 145 GPU-specific assertions with fallback validation

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
make gpu-usage-test                 # GPU performance test (alias)

# Clean (removes training outputs)
make clean
```

### Install Development Tools

```bash
make install-tools
```

## 📁 Project Structure

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
├── samples/              # Example programs
├── docs/                 # Documentation
└── README.md             # Main project README
```

## 📚 Advanced Usage

### Custom Training Callbacks

```cpp
// Training with progress monitoring and model saving
model.train(X, Y, loss, optimizer, 
    [&](int epoch, double current_loss) {
        if (epoch % 100 == 0) {
            printf("Epoch %d: Loss = %f\n", epoch, current_loss);
            
            // Save checkpoint
            std::string checkpoint = "models/epoch_" + std::to_string(epoch) + ".bin";
            model::ModelIO::save_model(model, checkpoint, model::ModelFormat::BINARY);
        }
    }, 
    1000  // Total epochs
);
```

### Model Architecture Customization

```cpp
// Build complex architecture
model::Sequential model;

// Input processing
model.add(std::make_shared<layer::Dense>(784, 256));  // Input layer
model.add(std::make_shared<layer::activation::ReLU>());

// Hidden layers
model.add(std::make_shared<layer::Dense>(256, 128));
model.add(std::make_shared<layer::activation::ReLU>());
model.add(std::make_shared<layer::Dense>(128, 64));
model.add(std::make_shared<layer::activation::Tanh>());

// Output layer
model.add(std::make_shared<layer::Dense>(64, 10));
model.add(std::make_shared<layer::activation::Sigmoid>());
```

### Error Handling

```cpp
try {
    // Model operations with error handling
    auto model = model::ModelIO::load_model("model.bin", model::ModelFormat::BINARY);
    model.train(X, Y, loss, optimizer);
} catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    // Handle error appropriately
}
```

## 🤝 Contributing

We welcome contributions to MLLib! Here's how you can help:

### Code Style

- Follow K&R formatting style
- Use meaningful variable names
- Add comprehensive unit tests for new features
- Update documentation for API changes

### Development Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with tests
4. Run quality checks (`make lint-all`)
5. Ensure all tests pass (`make test`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

### Testing Requirements

All contributions must include:
- Unit tests for new functionality
- Integration tests for complex features
- Performance tests where applicable
- Documentation updates

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.

## 🙏 Acknowledgments

- Modern C++17 design principles
- Comprehensive testing methodologies
- Performance-oriented implementation
- Cross-platform compatibility focus

---

For more detailed information, see our [documentation index](README.md) or explore the [samples directory](../samples/) for practical examples.ws/CI/badge.svg)](https://github.com/shadowlink0122/CppML/actions/workflows/ci.yml)
[![Extended CI](https://github.com/shadowlink0122/CppML/workflows/Extended%20CI/badge.svg)](https://github.com/shadowlink0122/CppML/actions/workflows/extended-ci.yml)
[![GPU CI](https://github.com/shadowlink0122/CppML/workflows/GPU%20CI/badge.svg)](https://github.com/shadowlink0122/CppML/actions/workflows/gpu-ci.yml)
[![Code Quality](https://img.shields.io/badge/code%20style-K%26R-blue.svg)](https://en.wikipedia.org/wiki/Indentation_style#K&R_style)
[![Tests](https://img.shields.io/badge/tests-21%2F21_unit_tests-brightgreen.svg)](#-testing)
[![Integration Tests](https://img.shields.io/badge/integration-3429%2F3429_assertions-brightgreen.svg)](#-testing)
[![GPU Tests](https://img.shields.io/badge/GPU_tests-145_assertions-blue.svg)](#-gpu-support)
[![Test Coverage](https://img.shields.io/badge/coverage-100%25_CI_success-brightgreen.svg)](#-testing)

A modern C++17 machine learning library designed for neural network training and model persistence with comprehensive testing and performance monitoring.

## ✨ Features

- **🧠 Neural Networks**: Sequential models with customizable architectures
- **� Layers**: Dense (fully connected), ReLU, Sigmoid, Tanh activation functions  
- **🎯 Training**: MSE loss function with SGD optimizer
- **🎯 Multi-GPU Support**: NVIDIA CUDA, AMD ROCm, Intel oneAPI, Apple Metal with automatic detection
- **💾 Model I/O**: Save/load models in binary, JSON, and config formats
- **📁 Auto Directory**: Automatic directory creation with `mkdir -p` functionality
- **🔧 Type Safety**: Enum-based format specification for improved reliability
- **⚡ Performance**: Optimized C++17 implementation with NDArray backend
- **🧪 Testing**: Comprehensive unit (21/21) and integration tests (3429/3429 assertions)
- **🖥️ GPU Testing**: 145 GPU-specific assertions with fallback validation
- **🔄 Cross-platform**: Linux, macOS, Windows support
- **📊 Benchmarking**: Real-time performance metrics and execution time tracking
- **� CI/CD Ready**: 100% test success rate for production deployments

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

## 🎯 Multi-GPU Support

MLLib provides comprehensive GPU acceleration across all major vendors:

### Supported GPU Vendors

| Vendor | Technology | Status | Auto-Detection |
|--------|------------|--------|----------------|
| **NVIDIA** | CUDA | ✅ Full Support | ✅ Automatic |
| **AMD** | ROCm/HIP | ✅ Full Support | ✅ Automatic |
| **Intel** | oneAPI/SYCL | ✅ Full Support | ✅ Automatic |
| **Apple** | Metal Performance Shaders | ✅ Full Support | ✅ Automatic |

### GPU Usage

```cpp
#include "MLLib.hpp"
using namespace MLLib;

// Enable GPU acceleration (automatic vendor detection)
Device::setDevice(DeviceType::GPU);

// Your training code runs on best available GPU
model::Sequential model;
// ... build your model ...
model.train(X, Y, loss, optimizer);
```

### GPU Performance Example

```bash
# Run GPU performance test
make gpu-usage-test

# Sample output on Apple Silicon M1:
# 🔸 Testing 1024x1024 matrix multiplication 🔸
# 🖥️  CPU time: 45000 μs (45.00 ms)
# 🚀 GPU time: 1800 μs (1.80 ms)
# 🎉 GPU is 25.0x faster than CPU! (Excellent acceleration)
```

### 🧪 Testing Documentation
- 🇺🇸 [Testing Guide (English)](TESTING_en.md)
- 🇯🇵 [テストガイド (日本語)](TESTING_ja.md)

### 📖 Language Versions
- 🇺🇸 [Documentation (English)](README_en.md)
- 🇯🇵 [ドキュメント (日本語)](README_ja.md)

## 📖 Overview

MLLib is a machine learning library written in C++17, featuring a comprehensive testing system with execution time monitoring.

### ✨ Key Features

- **Neural Networks**: Sequential models with multi-layer perceptrons
- **Layers**: Dense (fully connected), ReLU, Sigmoid, Tanh activation functions
- **Loss Functions**: Mean Squared Error (MSE)
- **Optimizers**: Stochastic Gradient Descent (SGD)
- **Model I/O**: Save/load models in binary, JSON, and config formats
- **Auto Directory Creation**: `mkdir -p` equivalent functionality
- **Type Safety**: Enum-based format specification system
- **Testing Framework**: Comprehensive test system with execution time monitoring
- **Performance Monitoring**: Microsecond precision execution time measurement

### 🧪 Testing System

MLLib provides a comprehensive testing framework:

```bash
# Unit tests (21/21 passing with timing display)
make unit-test

# Sample output:
# ✅ ConfigConstantsTest PASSED (10 assertions, 0.03ms)
# ✅ NDArrayMatmulTest PASSED (11 assertions, 0.02ms)
# Total test execution time: 0.45ms
# Total suite time (including overhead): 0.89ms

# Integration tests
make integration-test

# All tests
make test
```

#### Test Coverage

- **Config Module (3/3)**: Library configuration and constants
- **NDArray Module (6/6)**: Multi-dimensional array operations
- **Dense Layer (4/4)**: Fully connected layer functionality
- **Activation Functions (7/7)**: ReLU, Sigmoid, Tanh with error handling
- **Sequential Model (1/1)**: Model construction and prediction

#### Integration Test Coverage

- **XOR Problem**: End-to-end training and prediction
- **Model I/O**: Save/load workflows
- **Multi-layer**: Complex architecture validation
- **Performance**: Stability and performance testing

### 🏗️ Architecture

```
MLLib/
├── NDArray          # Multi-dimensional arrays (tensor operations)
├── Device           # CPU/GPU device management
├── Layer/           # Neural network layers
│   ├── Dense        # Fully connected layers
│   └── Activation/  # Activation functions
├── Loss/            # Loss functions
├── Optimizer/       # Optimization algorithms
└── Model/           # Model definition and I/O
```

### 🎯 Usage Example

```cpp
#include "MLLib/MLLib.hpp"

using namespace MLLib;

// XOR network example with epoch saving
auto model = std::make_unique<model::Sequential>();
model->add(std::make_shared<layer::Dense>(2, 4));
model->add(std::make_shared<layer::activation::ReLU>());  
model->add(std::make_shared<layer::Dense>(4, 1));
model->add(std::make_shared<layer::activation::Sigmoid>());

// Training data
std::vector<std::vector<double>> X = {{0,0}, {0,1}, {1,0}, {1,1}};
std::vector<std::vector<double>> Y = {{0}, {1}, {1}, {0}};

// Loss function and optimizer
loss::MSE loss_fn;
optimizer::SGD optimizer(0.1);

// Epoch saving callback
auto save_callback = [&model](int epoch, double loss) {
    if (epoch % 10 == 0) {
        std::cout << "Epoch " << epoch << ", Loss: " << loss << std::endl;
        
        // Type-safe enum format model saving
        std::string path = "models/training_xor/epoch_" + std::to_string(epoch);
        model::ModelIO::save_model(*model, path + ".bin", model::ModelFormat::BINARY);
        model::ModelIO::save_model(*model, path + ".json", model::ModelFormat::JSON);
    }
};

// Training
model->train(X, Y, loss_fn, optimizer, save_callback, 1000);

// Final model saving (all three formats)
model::ModelIO::save_model(*model, "models/final.bin", model::ModelFormat::BINARY);
model::ModelIO::save_model(*model, "models/final.json", model::ModelFormat::JSON);
model::ModelIO::save_model(*model, "models/final.config", model::ModelFormat::CONFIG);
```

## 🔧 Build and Test

```bash
# Build library
make

# Run samples
make samples               # Build all sample programs
make run-sample            # List available samples
make xor                   # XOR network execution (alias)
make device-detection      # GPU device detection sample (alias)
make gpu-vendor-detection  # GPU vendor detection sample (alias)

# Code quality checks
make fmt                    # Code formatting
make lint                   # Static analysis
make check                  # cppcheck execution
```

## 📁 Project Structure

```
CppML/
├── include/MLLib/         # Header files
├── src/MLLib/            # Implementation files
├── samples/              # Sample code
├── tests/                # Test code
├── docs/                 # Documentation
├── .github/workflows/    # CI/CD configuration
└── README.md             # Project overview
```

## 🚀 Getting Started

1. **Clone**: `git clone https://github.com/shadowlink0122/CppML.git`
2. **Build**: `make`
3. **Test**: `make test`
4. **Documentation**: See [Model I/O](MODEL_IO_en.md) for details

## 📝 License

This project is released under the MIT License.

## 🤝 Contributing

Pull requests, issue reports, and feature suggestions are welcome.

---

**Note**: This library is developed for educational and research purposes. Please perform thorough testing before production use.
