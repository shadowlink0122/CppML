# MLLib - C++ Machine Learning Library

[![CI](https://github.com/shadowlink0122/CppML/workflows/CI/badge.svg)](https://github.com/shadowlink0122/CppML/actions/workflows/ci.yml)
[![Extended CI](https://github.com/shadowlink0122/CppML/workflows/Extended%20CI/badge.svg)](https://github.com/shadowlink0122/CppML/actions/workflows/extended-ci.yml)
[![Code Quality](https://img.shields.io/badge/code%20style-K%26R-blue.svg)](https://en.wikipedia.org/wiki/Indentation_style#K&R_style)

> **Languages**: [English](README.md) | [日本語](README_ja.md)

A modern C++17 machine learning library designed for performance and ease of use.

## ✨ Features

- **Header-only and compiled components** - Flexible integration options
- **Modern C++17** - Leverages latest language features
- **Comprehensive ML tools** - Neural networks, data processing, optimizers
- **Cross-platform** - Linux, macOS, Windows support
- **CI/CD ready** - Automated testing and quality checks

## 🚀 Quick Start

### Prerequisites

- C++17 compatible compiler (GCC 7+, Clang 5+, MSVC 2017+)
- CMake 3.14+ (optional)
- Make

### Build

```bash
git clone https://github.com/shadowlink0122/CppML.git
cd CppML
make all
```

### Usage

```cpp
#include "MLLib.hpp"

int main() {
    MLLib::initialize();
    
    // Your ML code here
    
    MLLib::cleanup();
    return 0;
}
```

## 🛠️ Development

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
make all

# Debug build
make debug

# Build examples
make examples

# Run tests
make test

# Clean
make clean
```

### Install Development Tools

```bash
make install-tools
```

## 📚 Documentation

### API Components

- **Core**: `MLLib/config.hpp`, `MLLib/ndarray.hpp`
- **Data**: Preprocessing, batching, loading
- **Layers**: Dense, convolution, pooling, activation
- **Models**: Sequential, functional, custom
- **Optimizers**: SGD, Adam
- **Utils**: String, math, I/O, system utilities

### Example

```cpp
#include "MLLib.hpp"

// Create a simple neural network
MLLib::Sequential model;
model.add(new MLLib::Dense(128, 64));
model.add(new MLLib::ReLU());
model.add(new MLLib::Dense(64, 10));
model.compile(MLLib::Adam(), MLLib::CrossEntropy());
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run quality checks (`make lint-all`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Code Style

This project uses K&R style formatting. Please run `make fmt` before submitting.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by modern ML frameworks
- Built with modern C++ best practices
- Designed for performance and usability