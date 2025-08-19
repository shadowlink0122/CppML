# MLLib Model I/O with Directory Management

> **Language**: 🇺🇸 English | [🇯🇵 日本語](MODEL_IO_ja.md)

[![Model I/O Generalization](https://img.shields.io/badge/Model_I%2FO_generalization-100%25-green.svg)](#features)
[![Auto-Serialization](https://img.shields.io/badge/Auto_Serialization-Complete-blue.svg)](#usage-examples)
[![Binary Format](https://img.shields.io/badge/Binary_Format-Supported-orange.svg)](#features)
[![Generic Loading](https://img.shields.io/badge/Generic_Loading-Active-purple.svg)](#features)

This example demonstrates how to save and load models with automatic directory creation and 100% generalized Model I/O system with 76 unit tests coverage.

## Features

- **100% Model I/O Generalization**: Unified save/load system across all model types
- **Type-Safe Templates**: GenericModelIO::load_model<T> with runtime type checking
- **Large-Scale Model Support**: Tested up to 2048×2048 (4.2M parameters, 32MB)
- **High Performance**: Save 65ms, Load 163ms (large models)
- **Perfect Precision Weights/Biases**: Binary format preserves 1e-10 level precision
- **Automatic Directory Creation**: Directories are created automatically using `mkdir -p` equivalent functionality
- **Multiple File Formats**: BINARY, JSON, and CONFIG format support
- **8 Activation Functions**: ReLU, Sigmoid, Tanh, LeakyReLU, ELU, Swish, GELU, Softmax
- **76 Unit Tests**: 100% pass rate for quality assurance
- **Cross-Platform**: Works on Windows, macOS, and Linux

## 🔧 GenericModelIO Unified Interface

MLLib v1.0.0 supports type-safe GenericModelIO system:

```cpp
// GenericModelIO unified save (recommended)
GenericModelIO::save_model(*model, "model.bin", SaveFormat::BINARY);
GenericModelIO::save_model(*model, "model.json", SaveFormat::JSON);
GenericModelIO::save_model(*model, "model.config", SaveFormat::CONFIG);

// Type-safe loading (template function)
auto autoencoder = GenericModelIO::load_model<DenseAutoencoder>("model.bin", SaveFormat::BINARY);
auto sequential = GenericModelIO::load_model<Sequential>("model.bin", SaveFormat::BINARY);

// Large-scale model support (verified up to 2048x2048)
// File size: 32MB, Save: 65ms, Load: 163ms
```

## 🚀 Supported Model Types

### ✅ Full Support (76 Unit Tests Passing)

1. **DenseAutoencoder**
   - Encoder/Decoder structure
   - Custom activation functions
   - Complex architecture support

2. **Sequential**
   - 8 activation function types (ReLU, Sigmoid, Tanh, LeakyReLU, ELU, Swish, GELU, Softmax)
   - Parameterized activations (LeakyReLU alpha, ELU alpha)
   - Large networks (up to 4.2M parameters verified)

### 🔥 Large-Scale Model Performance

- **2048×2048 Model**: 4.2M parameters, 32MB
- **Save Time**: 65ms
- **Load Time**: 163ms
- **Parameter Preservation**: 100% (1e-10 precision)

## Usage Examples

```cpp
#include "MLLib.hpp"

using namespace MLLib;
using namespace MLLib::model;
using namespace MLLib::model::autoencoder;

// 1. DenseAutoencoder example
auto autoencoder = std::make_unique<DenseAutoencoder>();
// ... configure and train ...

// Save with GenericModelIO
GenericModelIO::save_model(*autoencoder, "models/autoencoder", SaveFormat::BINARY);

// Type-safe loading
auto loaded_autoencoder = GenericModelIO::load_model<DenseAutoencoder>(
    "models/autoencoder.bin", SaveFormat::BINARY);

// 2. Sequential model example  
auto sequential = std::make_unique<Sequential>();
sequential->add(std::make_shared<layer::Dense>(784, 128));
sequential->add(std::make_shared<layer::activation::ReLU>());
sequential->add(std::make_shared<layer::Dense>(128, 10));
sequential->add(std::make_shared<layer::activation::Softmax>());

// Save in multiple formats (auto directory creation)
GenericModelIO::save_model(*sequential, "models/trained/neural_networks/model.bin", SaveFormat::BINARY);
GenericModelIO::save_model(*sequential, "exports/json/model.json", SaveFormat::JSON);

// Load
auto loaded_sequential = GenericModelIO::load_model<Sequential>(
    "models/trained/neural_networks/model.bin", SaveFormat::BINARY);

// 3. Large model example (2048x2048)
auto large_model = std::make_unique<Sequential>();
large_model->add(std::make_shared<layer::Dense>(2048, 2048));
large_model->add(std::make_shared<layer::activation::ReLU>());

// High-performance save/load for large models
GenericModelIO::save_model(*large_model, "large_models/model_2048.bin", SaveFormat::BINARY);
// File size: ~32MB, Save time: ~65ms
```
```

## Directory Structure Example

After running the complete test, the following directory structure is created:

```
├── models/trained/neural_networks/
│   └── test_model.bin
├── exports/
│   ├── binary/
│   │   └── model_v1.bin
│   └── json/
│       └── model_v1.json
├── configs/architectures/
│   └── model_v1_config.txt
├── weights/checkpoints/
│   └── model_v1_params.bin
└── data/projects/ml_experiments/2024/august/neural_networks/xor_solver/
    └── final_model.bin
```

## Configuration File Format

Configuration files are saved in a human-readable YAML-like format:

```yaml
# MLLib Model Configuration
model_type: Sequential
version: 1.0.0
device: CPU
layers:
  - type: Dense
    input_size: 2
    output_size: 4
    use_bias: true
  - type: ReLU
  - type: Dense
    input_size: 4
    output_size: 1
    use_bias: true
  - type: Sigmoid
```

## File Formats

### 1. **BINARY Format (`.bin`)**
- **High-Speed Loading**: Optimized binary format for fast processing
- **Small File Size**: Efficient compression (Sequential: 372B, Autoencoder: 1108B)
- **Magic Number**: 0x4D4C4C47 for file verification
- **Version Management**: Future compatibility guarantee
- **High Precision**: 1e-10 level parameter preservation

### 2. **JSON Format (`.json`)**
- **Human Readable**: For debugging, verification, and editing
- **Complete Metadata**: Includes configuration, parameters, and type information
- **Cross-Platform**: Standard JSON format
- **Development Support**: Optimal for model analysis and debugging

### 3. **CONFIG Format (`.config`)**
- **Architecture Only**: Lightweight configuration without parameters
- **YAML-like Format**: Human editable
- **Template Usage**: Optimal for model structure reuse

## Performance Results

| Model Size | File Size | Save Time | Load Time |
|------------|-----------|-----------|-----------|
| 512→256→128 | 1.25 MB | 3ms | 6ms |
| 1024→512→256 | 5.01 MB | 11ms | 29ms |
| 2048→1024→512 | 20.01 MB | 36ms | 90ms |
| **2048×2048** | **32.02 MB** | **65ms** | **163ms** |

### Precision Verification
- **Parameter Preservation Rate**: 100% (perfect match in 32/32 samples)
- **Numerical Precision**: 1e-10 level high-precision preservation
- **Validation Tests**: All 76 unit tests passing

## Error Handling

- Automatic directory creation with proper error reporting
- File format validation
- Version compatibility checking
- Comprehensive error messages for troubleshooting

## Platform Support

- **macOS/Linux**: Uses `mkdir -p` system command
- **Windows**: Uses `mkdir` with appropriate error handling
- **C++17**: Uses `std::filesystem` when available for enhanced functionality
