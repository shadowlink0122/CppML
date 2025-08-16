# MLLib Documentation

Comprehensive documentation for the MLLib machine learning library.

## 📚 Documentation Index

### 🔧 Technical Documentation
- 🇺🇸 [Model I/O (English)](MODEL_IO_en.md)
- 🇯🇵 [モデル I/O (日本語)](MODEL_IO_ja.md)

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
