# MLLib Sample Programs

> **Language**: 🇺🇸 English | [🇯🇵 日本語](README_ja.md)

This directory contains sample programs demonstrating the capabilities of the MLLib machine learning library. Each sample showcases different aspects of the library, from basic device detection to advanced GPU-accelerated training.

## Available Samples

### 1. Device Detection (`device_detection.cpp`)

**Purpose**: Demonstrates GPU detection and device management capabilities.

**Features**:
- Detects available GPUs from different vendors (NVIDIA, AMD, Intel, Apple)
- Shows GPU specifications (memory, compute capability, API support)
- Tests primary GPU vendor detection
- Validates vendor availability
- Demonstrates device configuration and validation

**Usage**:
```bash
make samples
./build/samples/device_detection
```

**Sample Output**:
```
=== MLLib Device Detection Sample ===
This sample demonstrates GPU detection capabilities.

--- GPU Detection ---
Detected 1 GPU(s):
  GPU 1:
    Vendor: Apple
    Name: Apple M1 Pro
    Memory: 16384 MB
    Compute Capable: Yes
    API Support: Metal

--- Primary GPU Vendor ---
Primary GPU vendor: Apple
```

### 2. GPU Performance Test (`gpu_usage_test.cpp`)

**Purpose**: Benchmarks GPU vs CPU performance with matrix operations of varying sizes.

**Features**:
- Tests matrix multiplication performance across different sizes (256x256 to 2048x2048)
- Compares GPU and CPU execution times
- Calculates performance speedup ratios
- Validates numerical accuracy between GPU and CPU results
- Tests GPU backend selection and switching

**Usage**:
```bash
make samples
./build/samples/gpu_usage_test
```

**Key Performance Insights**:
- **Small matrices (256x256)**: CPU may be faster due to GPU overhead
- **Medium matrices (512x512)**: Performance begins to equalize
- **Large matrices (1024x1024+)**: GPU shows significant advantages
- **Metal backend**: Optimized for Apple Silicon, uses float32 precision

### 3. XOR Neural Network Training (`xor.cpp`)

**Purpose**: Demonstrates complete machine learning workflow with XOR problem.

**Features**:
- Builds a neural network with Dense layers and activation functions
- Uses GPU acceleration for training
- Implements training callbacks for progress monitoring
- Saves model checkpoints during training (every 10 epochs)
- Supports multiple model formats (Binary, JSON, Config)
- Demonstrates model inference after training

**Network Architecture**:
```
Input Layer:  2 neurons (XOR inputs)
Hidden Layer: 4 neurons + ReLU activation
Output Layer: 1 neuron + Sigmoid activation
```

**Usage**:
```bash
make samples
./build/samples/xor
```

**Training Output**:
The training process will save models in `samples/training_xor/` directory:
- `epoch_X.bin` / `epoch_X.json`: Models saved every 10 epochs
- `final_model.*`: Final trained model in multiple formats

**Expected Results**:
```
0,0 => ~0.0  (False XOR False = False)
0,1 => ~1.0  (False XOR True = True)
1,0 => ~1.0  (True XOR False = True)  
1,1 => ~0.0  (True XOR True = False)
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

This builds all `.cpp` files in the samples directory and places executables in `build/samples/`.

### Run Individual Samples

```bash
# Run specific sample
make run-sample SAMPLE=device_detection
make run-sample SAMPLE=gpu_usage_test
make run-sample SAMPLE=xor

# Or run directly
./build/samples/device_detection
./build/samples/gpu_usage_test
./build/samples/xor
```

### List Available Samples

```bash
make run-sample
```

## GPU Backend Support

The samples automatically detect and use the best available GPU backend:

| Platform | Primary Backend | Fallback |
|----------|----------------|----------|
| **Apple Silicon (M1/M2/M3)** | Metal Performance Shaders | CPU |
| **NVIDIA GPUs** | CUDA | CPU |
| **AMD GPUs** | ROCm/HIP | CPU |
| **Intel GPUs** | oneAPI/SYCL | CPU |

## Sample Output Files

### XOR Training Output (`training_xor/`)

The XOR sample generates training artifacts:

- **Binary Models** (`.bin`): Compact, fast-loading format
- **JSON Models** (`.json`): Human-readable, debuggable format  
- **Config Files** (`.config`): Model architecture configuration

These files can be used to:
- Resume training from checkpoints
- Analyze training progress
- Load pre-trained models in other applications

## Performance Expectations

### Device Detection
- **Execution Time**: < 1 second
- **Memory Usage**: Minimal
- **Purpose**: Quick hardware capability assessment

### GPU Performance Test
- **Small Matrices (256x256)**: CPU often faster (GPU overhead)
- **Medium Matrices (512x512)**: Performance parity
- **Large Matrices (1024x1024+)**: GPU shows 2-10x speedup
- **Memory Usage**: Scales with matrix size (O(n²))

### XOR Training
- **Training Time**: 1-10 seconds (150 epochs)
- **Convergence**: Usually achieves < 0.01 loss
- **GPU Speedup**: 2-5x over CPU (depending on hardware)
- **Memory Usage**: Minimal (small network)

## Troubleshooting

### Common Issues

1. **No GPU Detected**:
   ```
   No GPUs detected
   GPU Available: No
   ```
   **Solution**: Check GPU drivers and hardware compatibility.

2. **GPU Performance Slower Than CPU**:
   ```
   CPU is 1.5x faster than GPU
   ```
   **Solution**: This is normal for small matrices. Try larger sizes or check GPU backend selection.

3. **Build Failures**:
   ```
   ❌ Some samples failed to build
   ```
   **Solution**: Ensure MLLib library is built first with `make all`.

4. **Metal Precision Differences**:
   ```
   Max difference: 0.000123
   ```
   **Solution**: This is expected - Metal uses float32, CPU uses double64.

### Debug Information

For additional debugging, check:
```bash
# Verify library build
make all

# Check GPU capabilities  
make gpu-check

# View detailed build output
make samples VERBOSE=1
```

## Integration with Your Projects

These samples serve as templates for integrating MLLib into your own projects:

1. **Copy relevant sample code** as a starting point
2. **Modify network architecture** for your use case
3. **Adapt data loading** for your datasets  
4. **Customize training callbacks** for your monitoring needs
5. **Choose appropriate model formats** for your deployment

## Next Steps

After exploring these samples:

1. **Read the documentation** in `docs/` directory
2. **Run the test suite** with `make test`
3. **Explore GPU integration tests** with `make gpu-integration-test`
4. **Check out advanced examples** in the test directories

For more information, see:
- [GPU Strategy Documentation](../docs/GPU_STRATEGY_en.md)
- [Testing Documentation](../docs/TESTING_en.md)
- [Model I/O Documentation](../docs/MODEL_IO_en.md)
