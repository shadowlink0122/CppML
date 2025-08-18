# MLLib Sample Programs

> **Language**: 🇺🇸 English | [🇯🇵 日本語](README_ja.md)

This directory contains sample programs demonstrating the capabilities of the MLLib machine learning library. Each sample showcases different aspects of the library, from basic device detection to advanced GPU-accelerated training.

## Directory Structure

```
samples/
├── autoencoder/  # Autoencoder samples (basic, denoising, anomaly detection, VAE)
├── gpu/          # GPU-related samples (device detection, performance testing)
├── nn/           # Neural network samples (training, inference)
├── README.md     # This file
└── README_ja.md  # Japanese version
```

## Available Samples

### Autoencoder Samples (`autoencoder/`)

#### 1. Basic Autoencoder (`basic_autoencoder.cpp`)

**Purpose**: Demonstrates fundamental autoencoder concepts with data compression.

**Features**:
- Compresses 4D input data through a 2D bottleneck
- Learns structured data representations
- Shows compression ratios and reconstruction quality
- Saves trained models in multiple formats

**Usage**:
```bash
make run-sample SAMPLE=autoencoder/basic_autoencoder
```

#### 2. Denoising Autoencoder (`denoising_autoencoder.cpp`)

**Purpose**: Shows how to remove noise from 8x8 image patches.

**Features**:
- Trains on noisy input to clean output pairs
- Measures denoising quality with PSNR
- Visual comparison of original, noisy, and denoised images
- Achieves >10dB PSNR improvement

**Usage**:
```bash
make run-sample SAMPLE=autoencoder/denoising_autoencoder
```

#### 3. Anomaly Detection (`anomaly_detection.cpp`)

**Purpose**: Demonstrates using autoencoders for anomaly detection in sensor data.

**Features**:
- Trains only on normal sensor readings
- Detects various anomaly types (sensor failures, overheating, etc.)
- Calculates performance metrics (precision, recall, F1-score)
- Automatic threshold setting using percentiles

**Usage**:
```bash
make run-sample SAMPLE=autoencoder/anomaly_detection
```

#### 4. Variational Autoencoder Demo (`variational_autoencoder.cpp`)

**Purpose**: Introduces VAE concepts with a simplified implementation.

**Features**:
- Demonstrates probabilistic latent representations
- Works with 4x4 geometric patterns (circles, squares, crosses)
- Explains VAE theory and implementation requirements
- Shows generative modeling concepts

**Usage**:
```bash
make run-sample SAMPLE=autoencoder/variational_autoencoder
```

### GPU Samples (`gpu/`)

```

### GPU Samples (`gpu/`)

#### 1. Device Detection (`device_detection.cpp`)

**Purpose**: Demonstrates GPU detection and device management capabilities.

**Features**:
- Detects available GPUs from different vendors (NVIDIA, AMD, Intel, Apple)
- Shows GPU specifications (memory, compute capability, API support)
- Tests primary GPU vendor detection
- Validates vendor availability
- Demonstrates device configuration and validation

**Usage**:
```bash
make device-detection
# or
make run-sample SAMPLE=gpu/device_detection
```

#### 2. GPU Usage Test (`gpu_usage_test.cpp`)

**Purpose**: Benchmarks GPU vs CPU performance with matrix operations.

**Features**:
- Tests matrix multiplication performance across different sizes
- Compares GPU and CPU execution times
- Calculates performance speedup ratios
- Validates numerical accuracy between GPU and CPU results

**Usage**:
```bash
make run-sample SAMPLE=gpu/gpu_usage_test
```

#### 3. Debug GPU Detection (`debug_gpu_detection.cpp`)

**Purpose**: Comprehensive GPU detection debugging and system information.

**Features**:
- Detailed GPU vendor detection
- System profiler integration
- Hardware compatibility checking
- Debug output for troubleshooting

**Usage**:
```bash
make gpu-vendor-detection
# or
make run-sample SAMPLE=gpu/debug_gpu_detection
```

#### 4. Light GPU Detection (`debug_gpu_detection_light.cpp`)

**Purpose**: Lightweight GPU detection for quick system checks.

**Usage**:
```bash
make run-sample SAMPLE=gpu/debug_gpu_detection_light
```

#### 5. GPU Scenario Testing (`test_gpu_scenarios.cpp`, `verify_gpu_scenarios.cpp`)

**Purpose**: Test and verify various GPU usage scenarios.

**Usage**:
```bash
make run-sample SAMPLE=gpu/test_gpu_scenarios
make run-sample SAMPLE=gpu/verify_gpu_scenarios
```

### Neural Network Samples (`nn/`)

#### 1. XOR Neural Network (`xor.cpp`)

**Purpose**: Demonstrates complete machine learning workflow with XOR problem.

**Features**:
- Builds a neural network with Dense layers and activation functions
- Uses GPU acceleration for training
- Implements training callbacks for progress monitoring
- Saves model checkpoints during training (every 10 epochs)
- Supports multiple model formats (Binary, JSON, Config)

**Network Architecture**:
```
Input Layer:  2 neurons (XOR inputs)
Hidden Layer: 4 neurons + ReLU activation
Output Layer: 1 neuron + Sigmoid activation
```

**Usage**:
```bash
make xor
# or
make run-sample SAMPLE=nn/xor
```

## Building and Running Samples

### Prerequisites

1. **Build the MLLib library first**:
   ```bash
   make all
   ```

2. **Ensure GPU support is available** (if testing GPU samples):
   ```bash
   make gpu-check
   ```

### Build All Samples

```bash
make samples
```

This builds all samples and places executables in:
- `build/samples/gpu/` - GPU-related samples
- `build/samples/nn/` - Neural network samples

### Run Individual Samples

```bash
# Using aliases
make xor                    # Run XOR neural network sample
make device-detection       # Run device detection sample
make gpu-vendor-detection   # Run GPU vendor detection sample

# Using run-sample with full path
make run-sample SAMPLE=nn/xor
make run-sample SAMPLE=gpu/device_detection
make run-sample SAMPLE=gpu/gpu_usage_test
```

### List Available Samples

```bash
make run-sample
```

Output:
```
Usage: make run-sample SAMPLE=<sample_name>
       make run-sample SAMPLE=<subdir>/<sample_name>
Available samples:
  gpu/:
    gpu/debug_gpu_detection
    gpu/debug_gpu_detection_light
    gpu/device_detection
    gpu/gpu_usage_test
    gpu/test_gpu_scenarios
    gpu/verify_gpu_scenarios
  nn/:
    nn/xor
```

## GPU Backend Support

The samples automatically detect and use the best available GPU backend:

| Platform | Primary Backend | Fallback |
|----------|----------------|----------|
| **Apple Silicon (M1/M2/M3)** | Metal Performance Shaders | CPU |
| **NVIDIA GPUs** | CUDA | CPU |
| **AMD GPUs** | ROCm/HIP | CPU |
| **Intel GPUs** | oneAPI/SYCL | CPU |

## Sample Output Examples

### Device Detection Output
```
=== MLLib Device Detection Sample ===
--- GPU Detection ---
Detected 1 GPU(s):
  GPU 1:
    Vendor: Apple
    Name: Apple M1 GPU
    Memory: 12288 MB
    Compute Capable: Yes
    API Support: Metal

--- Primary GPU Vendor ---
Primary GPU vendor: Apple
```

### XOR Training Output
```
✅ GPU device successfully configured
Epoch 0 loss: 0.241477
Model saved at epoch 0 to samples/training_xor/epoch_0
Epoch 10 loss: 0.238898
...
0,0 => 0.468069  # Should approach 0.0
0,1 => 0.468069  # Should approach 1.0
1,0 => 0.648697  # Should approach 1.0
1,1 => 0.468069  # Should approach 0.0
```

## Performance Expectations

### Device Detection
- **Execution Time**: < 1 second
- **Memory Usage**: Minimal
- **Purpose**: Quick hardware capability assessment

### GPU Performance Test
- **Small matrices (256x256)**: CPU often faster (GPU overhead)
- **Medium matrices (512x512)**: Performance parity
- **Large matrices (1024x1024+)**: GPU shows 2-10x speedup
- **Memory Usage**: Scales with matrix size (O(n²))

### XOR Training
- **Training Time**: 1-10 seconds (150 epochs)
- **Convergence**: Usually achieves < 0.01 loss
- **GPU Speedup**: 2-5x over CPU (depending on hardware)
- **Memory Usage**: Minimal (small network)

## Output Files

### XOR Training Files (`samples/training_xor/`)
- **Binary Models** (`.bin`): Compact, fast-loading format
- **JSON Models** (`.json`): Human-readable, debuggable format
- **Config Files** (`.config`): Model architecture configuration

## Troubleshooting

### Common Issues

1. **No GPU Detected**:
   ```bash
   make run-sample SAMPLE=gpu/device_detection
   ```

2. **Build Failures**:
   ```bash
   make clean && make all && make samples
   ```

3. **GPU Performance Issues**:
   ```bash
   make run-sample SAMPLE=gpu/gpu_usage_test
   ```

### Debug Information

```bash
# Verify library build
make all

# Check GPU capabilities
make gpu-check

# List available samples
make run-sample
```

## Next Steps

After exploring these samples:

1. **Read the documentation** in `docs/` directory
2. **Run the test suite** with `make test`
3. **Explore GPU integration tests** with `make gpu-integration-test`
4. **Implement autoencoder examples** (coming soon in future updates)

For more information, see:
- [GPU Strategy Documentation](../docs/GPU_STRATEGY_en.md)
- [Testing Documentation](../docs/TESTING_en.md)
- [Model I/O Documentation](../docs/MODEL_IO_en.md)
