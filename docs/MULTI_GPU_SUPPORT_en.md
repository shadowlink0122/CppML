# Multi-GPU Support

CppML supports multiple GPU vendors and provides optimal GPU acceleration based on your system environment.

## Supported GPU Vendors

### 1. NVIDIA GPU (CUDA)
- **API**: CUDA Runtime API, cuBLAS
- **Supported Hardware**: GeForce, Quadro, Tesla, RTX series
- **Requirements**: 
  - CUDA Toolkit 11.0+
  - NVIDIA Driver 450.80.02+
  - Compute Capability 6.0+ recommended

### 2. AMD GPU (ROCm/HIP)
- **API**: ROCm, HIP, hipBLAS
- **Supported Hardware**: Radeon Instinct, Radeon Pro, select Radeon series
- **Requirements**:
  - ROCm 4.0+
  - Supported GPU (gfx900+)
  - Linux environment recommended

### 3. Intel GPU (oneAPI)
- **API**: Intel oneAPI, SYCL, oneMKL
- **Supported Hardware**: Intel Arc, Intel Iris Xe, Intel UHD Graphics
- **Requirements**:
  - Intel oneAPI Toolkit 2023.0+
  - Intel GPU Driver (latest recommended)
  - OpenCL 2.1+ support

### 4. Apple Silicon GPU (Metal)
- **API**: Metal Performance Shaders (MPS)
- **Supported Hardware**: M1, M1 Pro, M1 Max, M1 Ultra, M2 series
- **Requirements**:
  - macOS 12.0+
  - Xcode 13.0+
  - Metal 3 support

## Library Design Philosophy

### Default All-GPU Support
CppML is designed as a library with **all GPU vendors enabled by default**:

```makefile
# Enabled by default
CXXFLAGS += -DWITH_CUDA -DWITH_ROCM -DWITH_ONEAPI -DWITH_METAL
```

### Conditional Compilation
When GPU libraries are unavailable, the system automatically falls back to stub implementations:

```cpp
#ifdef WITH_CUDA
    // Actual CUDA implementation
#else
    // Stub implementation (CPU processing)
#endif
```

## Build Options

### Basic Build (Recommended)
```bash
# Build with all GPU support enabled
make clean && make
```

### Disabling Specific GPUs
You can disable specific GPU support as needed:

```bash
# Disable CUDA support
make DISABLE_CUDA=1

# Disable multiple GPU support
make DISABLE_CUDA=1 DISABLE_ROCM=1

# CPU-only build
make DISABLE_CUDA=1 DISABLE_ROCM=1 DISABLE_ONEAPI=1 DISABLE_METAL=1
```

### GPU Priority Order
When multiple GPUs are available, priority order is:
1. **NVIDIA CUDA** (highest priority)
2. **AMD ROCm**
3. **Intel oneAPI**
4. **Apple Metal**

## Usage

### Basic Usage Example
```cpp
#include "MLLib.hpp"

int main() {
    MLLib::model::Sequential model;
    
    // Specify GPU device (auto-detection & selection)
    model.set_device(MLLib::DeviceType::GPU);
    
    // Build layers
    model.add_layer(new MLLib::layer::Dense(784, 128));
    model.add_layer(new MLLib::layer::activation::ReLU());
    model.add_layer(new MLLib::layer::Dense(128, 10));
    
    // Execute training (automatic GPU utilization)
    model.train(train_X, train_Y, loss, optimizer);
    
    return 0;
}
```

### GPU Status Check
```cpp
// Check GPU availability
if (MLLib::Device::isGPUAvailable()) {
    printf("GPU available!\n");
    
    // Display available GPU details
    auto gpus = MLLib::Device::detectGPUs();
    for (const auto& gpu : gpus) {
        printf("GPU: %s (%s)\n", gpu.name.c_str(), gpu.api_support.c_str());
    }
}
```

## Warning System

### Automatic Warning Display
When GPU is requested but unavailable, the library automatically displays warnings:

```
⚠️  WARNING: GPU device requested but no GPU found!
   Falling back to CPU device for computation.
   Possible causes:
   - No NVIDIA GPU installed
   - CUDA driver not installed or incompatible
   - GPU is being used by another process
```

### Success Messages
When GPU is successfully configured:
```
✅ GPU device successfully configured
```

## Environment Variable Control

### Force CPU Execution
```bash
# Disable GPU for testing
FORCE_CPU_ONLY=1 ./your_program
```

### GPU Simulation
```bash
# Simulate GPU environment
GPU_SIMULATION_MODE=1 ./your_program
```

## Performance Optimization

### Memory Management
- **Unified Memory**: Uses CUDA Unified Memory (NVIDIA GPU)
- **Zero-Copy**: Leverages Apple Metal ManagedBuffer
- **Async Transfer**: ROCm/HIP asynchronous memory transfer

### Parallel Execution
```cpp
// Parallel execution in batch processing
model.train(large_dataset, batch_size=256);  // GPU parallel processing
```

## Troubleshooting

### CUDA Related
```bash
# Check CUDA information
nvidia-smi
nvcc --version

# Set CUDA environment variables
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

### ROCm Related
```bash
# Check ROCm information
rocm-smi
hipcc --version

# Set ROCm environment variables
export ROCM_PATH=/opt/rocm
export PATH=$ROCM_PATH/bin:$PATH
```

### oneAPI Related
```bash
# Initialize oneAPI environment
source /opt/intel/oneapi/setvars.sh
sycl-ls  # List SYCL devices
```

### Metal Related
```bash
# Check Metal capabilities
system_profiler SPDisplaysDataType
```

## Developer Information

### Adding New GPU Vendors
1. Add header file to `include/MLLib/backend/`
2. Add implementation file to `src/MLLib/backend/gpu/`
3. Add detection logic to `Device::detectGPUs()`
4. Add conditional compilation settings to Makefile

### Stub Implementations
To prevent compilation errors in GPU-unsupported environments, all GPU backends include stub implementations:

```cpp
#ifdef HIP_AVAILABLE
    // Actual ROCm implementation
    hipMalloc(&device_ptr, size);
#else
    // Stub implementation (CPU processing)
    device_ptr = malloc(size);
#endif
```

## Licenses and Notices

- **CUDA**: NVIDIA CUDA Toolkit End User License Agreement
- **ROCm**: MIT License
- **oneAPI**: Intel oneAPI License
- **Metal**: Apple Developer License

Please review each GPU vendor's license and documentation.

## Support and Community

- **Issues**: [GitHub Issues](https://github.com/shadowlink0122/CppML/issues)
- **Discussions**: [GitHub Discussions](https://github.com/shadowlink0122/CppML/discussions)
- **Wiki**: [CppML Wiki](https://github.com/shadowlink0122/CppML/wiki)

---

**Last Updated**: August 15, 2025  
**Compatible Version**: CppML v1.0.0+
