# GPU Detection Verification Guide

This guide helps verify that MLLib's GPU detection functionality works correctly across different Mac environments.

## Expected Results

### Apple Silicon Mac (M1/M2/M3)
```
Detected 1 GPU(s):
  GPU 1:
    Vendor: Apple
    Name: Apple M1 GPU (or M2/M3)
    Memory: 12288 MB (Unified memory - 75% of system RAM)
    API Support: Metal

Primary GPU vendor: Apple
```

### Intel Mac + AMD GPU (Example: Radeon Pro 5700)
```
Detected 1 GPU(s):
  GPU 1:
    Vendor: AMD
    Name: AMD Radeon Pro 5700
    Memory: 8192 MB (8GB dedicated VRAM)
    API Support: OpenCL/Metal

Primary GPU vendor: AMD
```

### Intel Mac + NVIDIA GPU (Example: GeForce RTX 3080)
```
Detected 1 GPU(s):
  GPU 1:
    Vendor: NVIDIA
    Name: NVIDIA GeForce RTX 3080
    Memory: 10240 MB (10GB dedicated VRAM)
    API Support: CUDA (if CUDA is installed)
    or API Support: OpenCL/Metal (without CUDA)

Primary GPU vendor: NVIDIA
```

### Intel Mac + Intel Integrated GPU (Example: Iris Plus Graphics)
```
Detected 1 GPU(s):
  GPU 1:
    Vendor: Intel
    Name: Intel Iris Plus Graphics
    Memory: 4096 MB (25% of system RAM shared)
    API Support: oneAPI/OpenCL

Primary GPU vendor: Intel
```

## Important Notes

### GPU Memory Detection Methods
MLLib uses different memory detection approaches based on GPU type:

#### Apple Silicon (M1/M2/M3)
- **Unified Memory Architecture**: CPU and GPU share system memory
- **Detection Method**: Get system memory via `sysctl hw.memsize`
- **GPU Available Memory**: Calculated as 75% of system memory
- **Benefits**: Large memory access, no memory copy overhead

#### AMD/NVIDIA Discrete GPUs
- **Dedicated VRAM**: GPU-specific graphics memory
- **Detection Method**: Parse VRAM info from `system_profiler SPDisplaysDataType`
- **Display Example**: "VRAM (Total): 8 GB" → detected as 8192 MB
- **Benefits**: High bandwidth, independent from CPU processing

#### Intel Integrated GPU
- **Shared Memory Architecture**: GPU uses part of system memory
- **Detection Method**: Conservatively calculated as 25% of system memory
- **Dynamic Allocation**: More memory available during actual usage

### Distinguishing Metal from GPU Vendor
- **Metal API Support ≠ Apple GPU Vendor**
- In Intel Mac + AMD GPU environments:
  - ✅ **Vendor**: AMD (correct - actual hardware vendor)
  - ✅ **API Support**: OpenCL/Metal (correct - API support on Mac)
  - ❌ **Vendor**: Apple (incorrect - not an Apple GPU)

### Common Misconceptions
```
Misconception: "Metal support means Apple GPU"
Reality: "AMD/NVIDIA GPUs also support Metal on macOS"

Misconception: "API Support: Metal → Vendor: Apple"
Reality: "AMD GPU + API Support: OpenCL/Metal → Vendor: AMD"
```

## Verification Steps

### 1. Check System Information
```bash
# Check architecture
sysctl hw.optional.arm64
# 1 = Apple Silicon, 0 or doesn't exist = Intel Mac

# Check GPU information
system_profiler SPDisplaysDataType | grep -A 5 "Chipset Model"
```

### 2. Run MLLib GPU Detection
```bash
cd CppML
make clean && make samples
make run-sample SAMPLE=device_detection
```

### 3. Detailed Debug Information (if needed)
```bash
# Run detailed test script
./test_gpu_detection.sh
```

## Issues That Need Reporting

Report issues in the following cases:

### ❌ Incorrect Detection Examples
- Intel Mac + AMD Radeon Pro → Vendor: Apple displayed
- Intel Mac + NVIDIA RTX → Vendor: Apple displayed
- Apple Silicon Mac → Vendor: AMD/NVIDIA displayed

### ✅ Normal Detection Examples
- Intel Mac + AMD Radeon Pro → Vendor: AMD, API Support: OpenCL/Metal
- Intel Mac + NVIDIA RTX → Vendor: NVIDIA, API Support: CUDA/OpenCL/Metal
- Apple Silicon Mac → Vendor: Apple, API Support: Metal

## Troubleshooting

### Intel Mac + Discrete GPU Shows "Apple"

1. Check system_profiler output:
```bash
system_profiler SPDisplaysDataType | grep -A 5 "Chipset Model"
```

2. Expected output example (AMD GPU):
```
Chipset Model: AMD Radeon Pro 5700
```

3. Expected output example (NVIDIA GPU):
```
Chipset Model: NVIDIA GeForce RTX 3080
```

4. Compare with actual detection results

### System Information Collection
When reporting issues, please include the following information:
```bash
# System information
uname -a
sysctl hw.model
sysctl hw.optional.arm64
system_profiler SPDisplaysDataType | grep -A 10 "Chipset Model\|Metal Support"

# MLLib detection results
make run-sample SAMPLE=device_detection
```

## Expected Behavior Technical Details

### GPU Detection Priority
1. **NVIDIA GPU** (discrete GPU, high performance)
2. **AMD GPU** (discrete GPU, high performance)
3. **Apple GPU** (Apple Silicon integrated GPU)
4. **Intel GPU** (Intel integrated GPU, power efficient)

### API Support Determination
- **CUDA**: NVIDIA GPU + CUDA driver installed
- **ROCm**: AMD GPU + ROCm installed
- **Metal**: Available for all GPUs on macOS (AMD, NVIDIA, Apple, Intel)
- **OpenCL**: Available for almost all GPUs

With this design, Intel Mac + AMD GPU environments will show:
- Vendor: AMD (correct hardware vendor)
- API Support: OpenCL/Metal (APIs available on macOS)

## Memory Detection Details

### Memory Architecture Types

#### Unified Memory (Apple Silicon)
- **Architecture**: CPU and GPU share the same physical memory pool
- **Advantages**: 
  - Zero-copy data sharing between CPU and GPU
  - Large memory capacity (up to system RAM)
  - Efficient for ML workloads with large datasets
- **Detection**: 75% of system RAM reported as GPU-accessible
- **Example**: 16GB system → 12,288 MB GPU memory

#### Dedicated VRAM (AMD/NVIDIA)
- **Architecture**: Separate high-bandwidth memory exclusively for GPU
- **Advantages**:
  - Very high memory bandwidth (up to 1TB/s)
  - Independent from CPU memory operations
  - Optimized for parallel GPU computations
- **Detection**: Parsed from system_profiler VRAM information
- **Example**: RTX 3080 → 10,240 MB dedicated VRAM

#### Shared Memory (Intel Integrated)
- **Architecture**: GPU dynamically borrows from system RAM
- **Advantages**:
  - Power efficient
  - Good for lightweight GPU tasks
  - No additional memory cost
- **Detection**: Conservative estimate of 25% system RAM
- **Example**: 16GB system → 4,096 MB shared memory

### Memory Usage Recommendations

#### For Large Models (>4GB)
- **Apple Silicon**: Excellent - unified memory allows large models
- **Discrete GPU**: Check VRAM capacity vs model size
- **Intel Integrated**: May require model optimization or CPU fallback

#### For Training Workloads
- **Apple Silicon**: Good - efficient CPU-GPU data sharing
- **Discrete GPU**: Excellent - high bandwidth for gradients
- **Intel Integrated**: Limited - best for inference only

#### For Inference Only
- **All GPU Types**: Generally suitable with appropriate batch sizes

## Testing Scenarios

### Scenario 1: Apple Silicon Development
```bash
# Expected on M1/M2/M3 Macs
Vendor: Apple
Memory: 8192-24576 MB (depends on system RAM)
API Support: Metal
Performance: Excellent for unified memory workloads
```

### Scenario 2: Intel Mac Workstation
```bash
# Expected with discrete GPU
Vendor: AMD or NVIDIA (not Apple)
Memory: 4096-24576 MB (depends on GPU model)
API Support: Metal/OpenCL or CUDA
Performance: Excellent for compute-intensive tasks
```

### Scenario 3: Intel Mac Laptop
```bash
# Expected with integrated graphics
Vendor: Intel
Memory: 1024-4096 MB (shared with system)
API Support: OpenCL/Metal
Performance: Good for light ML tasks
```

## Advanced Verification

### Manual System Profiler Check
```bash
# Get detailed GPU information
system_profiler SPDisplaysDataType

# Check for specific vendors
system_profiler SPDisplaysDataType | grep -i "AMD\|NVIDIA\|Intel\|Apple"

# Check memory information
system_profiler SPDisplaysDataType | grep -i "VRAM\|Memory"
```

### Architecture Detection
```bash
# Check if Apple Silicon
if [ "$(sysctl -n hw.optional.arm64 2>/dev/null)" = "1" ]; then
    echo "Apple Silicon - expect Apple GPU"
else
    echo "Intel Mac - expect discrete or Intel integrated GPU"
fi
```

### Metal Framework Availability
```bash
# Check Metal support (should be available on all modern Macs)
system_profiler SPDisplaysDataType | grep -i "Metal Support"
```

This comprehensive verification ensures that MLLib correctly identifies GPU hardware vendors while properly supporting Metal API across all Mac configurations.
