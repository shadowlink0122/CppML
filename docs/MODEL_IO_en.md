# MLLib Model I/O with Directory Management

> **Language**: 🇺🇸 English | [🇯🇵 日本語](MODEL_IO_ja.md)

[![Model I/O Generalization](https://img.shields.io/badge/Model_I%2FO_generalization-92%25-green.svg)](#features)
[![Auto-Serialization](https://img.shields.io/badge/Auto_Serialization-Complete-blue.svg)](#usage-examples)
[![Binary Format](https://img.shields.io/badge/Binary_Format-Supported-orange.svg)](#features)
[![Generic Loading](https://img.shields.io/badge/Generic_Loading-Active-purple.svg)](#features)

This example demonstrates how to save and load models with automatic directory creation and 92% generalized Model I/O system.

## Features

- **92% Model I/O Generalization**: Unified save/load system across all model types
- **Auto-Serialization**: Automatic model serialization with type safety
- **Generic Model Loading**: Template-based loading for any model architecture
- **Perfect Precision Weights/Biases**: Binary format preserves exact numerical values
- **Automatic Directory Creation**: Directories are created automatically using `mkdir -p` equivalent functionality
- **Multiple File Formats**: Support for binary, JSON, and configuration formats
- **Nested Directory Support**: Can handle deep nested paths
- **Cross-Platform**: Works on Windows, macOS, and Linux

## Usage Examples

```cpp
#include "MLLib/MLLib.hpp"

using namespace MLLib;

// Create a model
auto model = std::make_unique<model::Sequential>();
model->add(std::make_shared<layer::Dense>(2, 4, true));
model->add(std::make_shared<layer::activation::ReLU>());
model->add(std::make_shared<layer::Dense>(4, 1, true));
model->add(std::make_shared<layer::activation::Sigmoid>());

// Save complete model to nested directory (auto-creates directories)
model::ModelIO::save_model(*model, "models/trained/neural_networks/my_model.bin", "binary");

// Save different components to organized directories
model::ModelIO::save_config(*model, "configs/architectures/my_model_config.txt");
model::ModelIO::save_parameters(*model, "weights/checkpoints/my_model_params.bin");

// Save in different formats
model::ModelIO::save_model(*model, "exports/binary/my_model.bin", "binary");
model::ModelIO::save_model(*model, "exports/json/my_model.json", "json");

// Load from nested directories
auto loaded_model = model::ModelIO::load_model("models/trained/neural_networks/my_model.bin", "binary");

// Load architecture and parameters separately
auto config_model = model::ModelIO::load_config("configs/architectures/my_model_config.txt");
model::ModelIO::load_parameters(*config_model, "weights/checkpoints/my_model_params.bin");
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

1. **Binary Format (`.bin`)**: Efficient storage with magic numbers and versioning
2. **JSON Format (`.json`)**: Human-readable format with complete model information
3. **Configuration Format (`.txt`)**: Architecture-only information in YAML-like format
4. **Parameters Format (`.bin`)**: Weights and biases only in binary format

## Error Handling

- Automatic directory creation with proper error reporting
- File format validation
- Version compatibility checking
- Comprehensive error messages for troubleshooting

## Platform Support

- **macOS/Linux**: Uses `mkdir -p` system command
- **Windows**: Uses `mkdir` with appropriate error handling
- **C++17**: Uses `std::filesystem` when available for enhanced functionality
