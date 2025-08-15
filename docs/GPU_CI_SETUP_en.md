# GPU Testing Environment Setup

This document explains how to set up GPU environment testing in CI.

## Overview

MLLib's GPU implementation adopts the following testing strategy:

1. **CPU Fallback Testing**: Test GPU alternative behavior on CPU
2. **GPU Simulation Testing**: Test with simulated CUDA environment
3. **GPU Hardware Testing**: Test on actual GPU hardware (optional)

## CI Configuration

### 1. Automatically Executed Tests

The following tests are automatically executed by `.github/workflows/gpu-ci.yml`:

- **GPU Fallback Test**: Behavior testing in GPU-disabled environment
- **GPU Simulation Test**: Testing in CUDA simulation environment
- **GPU Documentation Check**: Verification of GPU-related documentation

### 2. Manually Executable Tests

Can be manually executed from GitHub Actions Workflow Dispatch:

```bash
# Test type selection:
# - all: All GPU tests
# - unit: GPU unit tests only
# - integration: GPU integration tests only
# - benchmark: GPU benchmarks only
```

### 3. GPU Hardware Testing (Optional)

To run tests on actual GPU hardware:

#### 3.1 Self-hosted Runner Setup

1. Set up self-hosted runner on machine with NVIDIA GPU
2. Install CUDA toolkit
3. Set `self-hosted-gpu` label on runner

#### 3.2 Workflow Activation

Modify the following in `.github/workflows/gpu-ci.yml`:

```yaml
# Before:
runs-on: ubuntu-latest  # Change to 'self-hosted-gpu' when GPU runner is available
if: false  # Change to 'needs.gpu-availability-check.outputs.gpu-available == true' when ready

# After:
runs-on: self-hosted-gpu
if: needs.gpu-availability-check.outputs.gpu-available == 'true'
```

#### 3.3 Environment Variable Configuration

Modify GPU environment check section:

```yaml
# In gpu-availability-check job:
if [ -n "${{ secrets.GITHUB_TOKEN }}" ]; then
  echo "gpu-available=true" >> $GITHUB_OUTPUT  # Change to true
  echo "✅ GPU runners available"
```

## Test Content

### GPU Unit Tests (74 assertions)

- **GPUAvailabilityTest**: GPU detection functionality
- **GPUDeviceValidationTest**: GPU validation system
- **GPUBackendOperationsTest**: GPU basic operations
- **GPUMemoryManagementTest**: GPU memory management
- **GPUErrorHandlingTest**: GPU error handling
- **GPUPerformanceTest**: GPU performance measurement

### GPU Integration Tests (71 assertions)

- **GPUCPUFallbackIntegrationTest**: GPU-CPU fallback integration
- **GPUModelComplexityIntegrationTest**: GPU model complexity
- **GPUPerformanceIntegrationTest**: GPU performance integration
- **GPUMemoryIntegrationTest**: GPU memory integration

## GPU Features

### Implemented Features

1. **Automatic GPU Detection**: Automatic GPU environment detection at startup
2. **CPU Fallback**: Automatic CPU fallback when GPU unavailable
3. **Warning System**: User notification when GPU issues occur
4. **CUDA Integration**: Complete integration of CUDA kernels and cuBLAS
5. **Memory Management**: Efficient GPU/CPU memory management

### Tested Operations

- Matrix multiplication
- Element-wise operations
- Activation functions
- Memory transfer operations

## GPU Testing in Local Development Environment

### CUDA Environment Setup

```bash
# CUDA toolkit installation (Ubuntu)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-2

# Environment variable setup
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### GPU Test Execution

```bash
# Build with GPU enabled
make clean
make

# Run GPU tests (runtime detection)
make unit-test
make integration-test

# GPU environment check
nvidia-smi
nvcc --version
```

### CPU Fallback Testing

```bash
# Build with CUDA disabled
make clean
make

# Run fallback tests
make unit-test
make integration-test
```

## Troubleshooting

### Common Issues

1. **CUDA not found error**
   - Check CUDA environment in `cuda.mk`
   - Verify `nvcc` is in PATH

2. **GPU tests failing**
   - Verify GPU drivers are correctly installed
   - Check CUDA version compatibility

3. **Fallback tests failing**
   - Verify CPU-only build works correctly
   - Check warning messages are displayed properly

### Debug Commands

```bash
# CUDA environment check
which nvcc
nvidia-smi
cat /proc/driver/nvidia/version

# Detailed build check
make clean
make -d

# Detailed test execution
make unit-test VERBOSE=1
```

## Performance Metrics

### Expected Performance

- **GPU Tests**: ~10-15 seconds (hardware dependent)
- **CPU Fallback Tests**: ~5-10 seconds
- **Simulation Tests**: ~3-5 seconds

### Benchmark Standards

- Matrix multiplication (1000x1000): GPU < 10ms, CPU < 100ms
- Element-wise operations: GPU speedup 2-10x vs CPU
- Memory transfer overhead: < 5% of computation time

## CI Optimization

### Performance Improvements

1. **Parallel Execution**: Parallelize GPU/CPU tests
2. **Cache Utilization**: CUDA toolkit installation cache
3. **Early Exit**: Early termination on critical test failures

### Resource Management

1. **GPU Time Limits**: Limit long-running executions
2. **Memory Monitoring**: GPU memory usage monitoring
3. **Temperature Checks**: GPU temperature monitoring (when available)

## Summary

The GPU CI environment provides the following benefits:

- ✅ **Comprehensive Testing**: Tests both GPU and CPU execution paths
- ✅ **Automatic Verification**: Automatic GPU functionality verification on code changes
- ✅ **Quality Assurance**: Ensures GPU implementation reliability
- ✅ **Performance Monitoring**: Continuous GPU performance monitoring
- ✅ **Compatibility Verification**: Operation verification in various environments

This configuration ensures continuous quality and reliability of GPU implementation, enabling stable operation in production environments!
