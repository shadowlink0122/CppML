# MLLib - C++ Machine Learning Library

[![CI](https://github.com/shadowlink0122/CppML/workflows/CI/badge.svg)](https://github.com/shadowlink0122/CppML/actions/workflows/ci.yml)
[![Extended CI](https://github.com/shadowlink0122/CppML/workflows/Extended%20CI/badge.svg)](https://github.com/shadowlink0122/CppML/actions/workflows/extended-ci.yml)
[![Code Quality](https://img.shields.io/badge/code%20style-K%26R-blue.svg)](https://en.wikipedia.org/wiki/Indentation_style#K&R_style)
[![Tests](https://img.shields.io/badge/tests-21%2F21_passing-brightgreen.svg)](#-testing)

> **Languages**: [English](README.md) | [日本語](docs/README_ja.md)

A modern C++17 machine learning library designed for neural network training and model persistence with comprehensive testing and performance monitoring.

## ✨ Features

- **🧠 Neural Networks**: Sequential models with customizable architectures
- **📊 Layers**: Dense (fully connected), ReLU, Sigmoid, Tanh activation functions
- **🎯 Training**: MSE loss function with SGD optimizer
- **💾 Model I/O**: Save/load models in binary, JSON, and config formats
- **📁 Auto Directory**: Automatic directory creation with `mkdir -p` functionality
- **🔧 Type Safety**: Enum-based format specification for improved reliability
- **⚡ Performance**: Optimized C++17 implementation with NDArray backend
- **🧪 Testing**: Comprehensive unit and integration tests with execution time monitoring
- **🔄 Cross-platform**: Linux, macOS, Windows support
- **📊 Benchmarking**: Real-time performance metrics and execution time tracking

## 🚀 Quick Start

### Prerequisites

- C++17 compatible compiler (GCC 7+, Clang 5+, MSVC 2017+)
- Make

### Build and Test

```bash
git clone https://github.com/shadowlink0122/CppML.git
cd CppML
make                    # Build the library
make unit-test         # Run unit tests (21/21 passing)
make integration-test  # Run integration tests
make xor              # Run XOR neural network example
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
    std::cout << "Epoch " << epoch << ", Loss: " << loss << std::endl;
}, 1000);

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
make integration-test

# Tests include:
# - XOR problem end-to-end training
# - Model I/O workflows (save/load)
# - Multi-layer architecture validation
# - Performance and stability testing
```

## 🛠️ Development

### Testing

```bash
# Run all tests
make test                   # Unit + Integration tests
make unit-test             # Run unit tests (21/21 passing)
make integration-test      # End-to-end integration tests

# Test output includes execution time monitoring:
# ✅ NDArrayConstructorTest PASSED (14 assertions, 0.01ms)
# Total test execution time: 0.45ms
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
make xor                    # XOR neural network
make model-format-test      # Model I/O format testing

# Clean (removes training outputs)
make clean
```

### Install Development Tools

```bash
make install-tools
```

## 📚 Documentation

Comprehensive documentation is available in the [`docs/`](docs/) directory:

- **📖 [Overview](docs/README.md)** - Documentation index and quick start
- **🇺🇸 [English Documentation](docs/README_en.md)** - Complete guide in English
- **🇯🇵 [日本語ドキュメント](docs/README_ja.md)** - 日本語での完全ガイド
- **🇺🇸 [Model I/O Guide (English)](docs/MODEL_IO_en.md)** - Complete model serialization guide
- **🇯🇵 [モデル I/O ガイド (日本語)](docs/MODEL_IO_ja.md)** - 日本語でのモデル保存・読み込み解説
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

Test all model I/O formats:

```bash
make model-format-test  # Test enum-based format system
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
