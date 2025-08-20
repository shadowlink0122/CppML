#include "../../../include/MLLib/device/device.hpp"
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#ifdef WITH_CUDA
#include "../backend/gpu/cuda_kernels.hpp"
#endif

#ifdef __APPLE__
#include <sys/sysctl.h>
#endif

#ifdef __linux__
#include <unistd.h>
#endif

namespace MLLib {

// Forward declarations for helper functions
namespace {
bool checkNVIDIAGPU();
std::string detectNVIDIAGPUName();
int detectNVIDIAGPUMemoryMB();
bool checkAMDGPU();
std::string detectAMDGPUName();
int detectAMDGPUMemoryMB();
bool checkIntelGPU();
std::string detectIntelGPUName();
int detectIntelGPUMemoryMB();
#ifdef __APPLE__
bool checkAppleGPU();
std::string detectAppleGPUName();
int detectAppleGPUMemoryMB();
#endif
}  // namespace

// Static member definition
DeviceType Device::current_device_ = DeviceType::CPU;

bool Device::isGPUAvailable() {
#ifdef WITH_CUDA
  try {
    if (Backend::cuda::cuda_is_available()) {
      return true;
    }
  } catch (const std::exception& e) {
    // CUDA failed, continue checking other vendors
  }
#endif

  // Check for other GPU vendors
  auto gpus = Device::detectGPUs();
  return !gpus.empty();
}

std::vector<GPUInfo> Device::detectGPUs() {
  std::vector<GPUInfo> gpus;

// Debug: Log detection process in debug builds
#ifdef DEBUG_GPU_DETECTION
  printf("🔍 Starting GPU detection process...\n");
#endif

  // 1. Check NVIDIA GPUs (CUDA) first
  bool has_nvidia = checkNVIDIAGPU();
#ifdef DEBUG_GPU_DETECTION
  printf("🔍 NVIDIA check: %s\n", has_nvidia ? "Found" : "Not found");
#endif
  if (has_nvidia) {
    GPUInfo nvidia_gpu;
    nvidia_gpu.vendor = GPUVendor::NVIDIA;
    nvidia_gpu.name = detectNVIDIAGPUName();
    nvidia_gpu.memory_mb = detectNVIDIAGPUMemoryMB();
    nvidia_gpu.compute_capable = true;
#ifdef WITH_CUDA
    nvidia_gpu.api_support = "CUDA";
#else
    nvidia_gpu.api_support = "OpenCL/Metal";
#endif
    gpus.push_back(nvidia_gpu);
#ifdef DEBUG_GPU_DETECTION
    printf("✅ Added NVIDIA GPU: %s (%d MB)\n", nvidia_gpu.name.c_str(),
           nvidia_gpu.memory_mb);
#endif
  }

  // 2. Check AMD GPUs (ROCm/OpenCL)
  bool has_amd = checkAMDGPU();
#ifdef DEBUG_GPU_DETECTION
  printf("🔍 AMD check: %s\n", has_amd ? "Found" : "Not found");
#endif
  if (has_amd) {
    GPUInfo amd_gpu;
    amd_gpu.vendor = GPUVendor::AMD;
    amd_gpu.name = detectAMDGPUName();
    amd_gpu.memory_mb = detectAMDGPUMemoryMB();
    amd_gpu.compute_capable = true;
#ifdef WITH_ROCM
    amd_gpu.api_support = "ROCm";
#else
    amd_gpu.api_support = "OpenCL/Metal";
#endif
    gpus.push_back(amd_gpu);
#ifdef DEBUG_GPU_DETECTION
    printf("✅ Added AMD GPU: %s (%d MB)\n", amd_gpu.name.c_str(),
           amd_gpu.memory_mb);
#endif
  }

  // 3. Check Intel GPUs (oneAPI/OpenCL)
  bool has_intel = checkIntelGPU();
#ifdef DEBUG_GPU_DETECTION
  printf("🔍 Intel check: %s\n", has_intel ? "Found" : "Not found");
#endif
  if (has_intel) {
    GPUInfo intel_gpu;
    intel_gpu.vendor = GPUVendor::INTEL_GPU;
    intel_gpu.name = detectIntelGPUName();
    intel_gpu.memory_mb = detectIntelGPUMemoryMB();
    intel_gpu.compute_capable = true;
#ifdef WITH_ONEAPI
    intel_gpu.api_support = "oneAPI";
#else
    intel_gpu.api_support = "oneAPI/OpenCL";
#endif
    gpus.push_back(intel_gpu);
#ifdef DEBUG_GPU_DETECTION
    printf("✅ Added Intel GPU: %s (%d MB)\n", intel_gpu.name.c_str(),
           intel_gpu.memory_mb);
#endif
  }

  // 4. Check Apple Silicon GPUs (Metal) - only on Apple Silicon
#ifdef __APPLE__
  bool has_apple = checkAppleGPU();
#ifdef DEBUG_GPU_DETECTION
  printf("🔍 Apple check: %s\n", has_apple ? "Found" : "Not found");
#endif
  if (has_apple) {
    GPUInfo apple_gpu;
    apple_gpu.vendor = GPUVendor::APPLE;
    apple_gpu.name = detectAppleGPUName();
    apple_gpu.memory_mb = detectAppleGPUMemoryMB();
    apple_gpu.compute_capable = true;
    apple_gpu.api_support = "Metal";
    gpus.push_back(apple_gpu);
#ifdef DEBUG_GPU_DETECTION
    printf("✅ Added Apple GPU: %s (%d MB)\n", apple_gpu.name.c_str(),
           apple_gpu.memory_mb);
#endif
  }
#endif

#ifdef DEBUG_GPU_DETECTION
  printf("🔍 Total GPUs detected: %zu\n", gpus.size());
#endif

  return gpus;
}

GPUVendor Device::getPrimaryGPUVendor() {
  auto gpus = Device::detectGPUs();
  if (gpus.empty()) {
    return GPUVendor::UNKNOWN;
  }

  // Priority order: NVIDIA > AMD > Apple > Intel
  // This ensures discrete GPUs are preferred over integrated ones
  for (const auto& gpu : gpus) {
    if (gpu.vendor == GPUVendor::NVIDIA) return GPUVendor::NVIDIA;
  }
  for (const auto& gpu : gpus) {
    if (gpu.vendor == GPUVendor::AMD) return GPUVendor::AMD;
  }
  for (const auto& gpu : gpus) {
    if (gpu.vendor == GPUVendor::APPLE) return GPUVendor::APPLE;
  }
  for (const auto& gpu : gpus) {
    if (gpu.vendor == GPUVendor::INTEL_GPU) return GPUVendor::INTEL_GPU;
  }

  return gpus[0].vendor;
}

bool Device::isGPUVendorAvailable(GPUVendor vendor) {
  auto gpus = Device::detectGPUs();
  for (const auto& gpu : gpus) {
    if (gpu.vendor == vendor) {
      return true;
    }
  }
  return false;
}

bool Device::setDeviceWithValidation(DeviceType device, bool show_warnings) {
  if (device == DeviceType::GPU) {
    if (!isGPUAvailable()) {
      if (show_warnings) {
        printf("⚠️  WARNING: GPU device requested but no GPU found!\n");
        printf("   Falling back to CPU device for computation.\n");

#ifndef WITH_CUDA
        printf("   Note: MLLib was compiled without CUDA support.\n");
        printf(
            "   To enable GPU support, install CUDA and rebuild with: make clean && make all\n");
#else
        printf("   Possible causes:\n");
        printf("   - No NVIDIA GPU installed\n");
        printf("   - CUDA driver not installed or incompatible\n");
        printf("   - GPU is being used by another process\n");
#endif
      }
      current_device_ = DeviceType::CPU;
      return false;
    } else {
      if (show_warnings) {
        printf("✅ GPU device successfully configured\n");
      }
    }
  }

  current_device_ = device;
  return true;
}

const char* Device::getDeviceTypeString(DeviceType device) {
  switch (device) {
  case DeviceType::CPU: return "CPU";
  case DeviceType::GPU: return "GPU";
  case DeviceType::AUTO: return "AUTO";
  default: return "UNKNOWN";
  }
}

// Helper functions for GPU vendor detection with caching
namespace {

// Cache for GPU detection results
struct GPUDetectionCache {
  bool initialized = false;
  bool nvidia_available = false;
  bool amd_available = false;
  bool intel_available = false;
  std::string system_profiler_output;
  
  void initialize() {
    if (initialized) return;
    
#ifdef __APPLE__
    // Get system profiler output once
    FILE* pipe = popen("system_profiler SPDisplaysDataType 2>/dev/null", "r");
    if (pipe) {
      char buffer[4096];
      while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        system_profiler_output += buffer;
      }
      pclose(pipe);
      
      // Check for different GPU vendors in cached output
      std::string lower_output = system_profiler_output;
      std::transform(lower_output.begin(), lower_output.end(), lower_output.begin(), ::tolower);
      
      nvidia_available = (lower_output.find("nvidia") != std::string::npos ||
                         lower_output.find("geforce") != std::string::npos ||
                         lower_output.find("quadro") != std::string::npos);
      
      amd_available = (lower_output.find("amd") != std::string::npos ||
                      lower_output.find("radeon") != std::string::npos);
      
      intel_available = (lower_output.find("intel") != std::string::npos);
    }
#endif
    initialized = true;
  }
};

static GPUDetectionCache gpu_cache;

bool checkNVIDIAGPU() {
  gpu_cache.initialize();
#ifdef __APPLE__
  return gpu_cache.nvidia_available;
#endif

#ifdef __linux__
  // Check for NVIDIA GPU via nvidia-smi or /proc
  FILE* pipe = popen("nvidia-smi -L 2>/dev/null", "r");
  if (pipe) {
    char buffer[256];
    std::string result;
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
      result += buffer;
    }
    pclose(pipe);

    if (!result.empty() && result.find("GPU") != std::string::npos) {
      return true;
    }
  }

  // Check for NVIDIA driver
  std::ifstream proc_modules("/proc/modules");
  if (proc_modules.is_open()) {
    std::string line;
    while (std::getline(proc_modules, line)) {
      if (line.find("nvidia") != std::string::npos) {
        return true;
      }
    }
  }
#endif

#ifdef _WIN32
  // Windows: Check via nvidia-smi or WMI
  // TODO: Implement Windows NVIDIA GPU detection
#endif

  return false;
}

std::string detectNVIDIAGPUName() {
#ifdef __APPLE__
  // Get NVIDIA GPU name from system_profiler on Mac
  FILE* pipe = popen(
      "system_profiler SPDisplaysDataType | grep -i 'Chipset Model' | grep -i 'nvidia\\|geforce\\|quadro' | head -1 2>/dev/null",
      "r");
  if (pipe) {
    char buffer[256];
    std::string result;
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
      result += buffer;
    }
    pclose(pipe);

    if (!result.empty()) {
      // Extract GPU name from system_profiler output
      size_t pos = result.find(":");
      if (pos != std::string::npos) {
        std::string gpu_name = result.substr(pos + 1);
        // Remove whitespace and newlines
        gpu_name.erase(0, gpu_name.find_first_not_of(" \t\n\r"));
        gpu_name.erase(gpu_name.find_last_not_of(" \t\n\r") + 1);
        return gpu_name;
      }
    }
  }
#endif

#ifdef __linux__
  // Try to get GPU name from nvidia-smi
  FILE* pipe = popen(
      "nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null",
      "r");
  if (pipe) {
    char buffer[256];
    std::string result;
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
      result += buffer;
    }
    pclose(pipe);

    if (!result.empty()) {
      // Remove whitespace and newlines
      result.erase(0, result.find_first_not_of(" \t\n\r"));
      result.erase(result.find_last_not_of(" \t\n\r") + 1);
      return result;
    }
  }

  // Fallback to generic name
  return "NVIDIA GPU";
#endif
  return "NVIDIA GPU";
}

bool checkAMDGPU() {
  gpu_cache.initialize();
#ifdef __APPLE__
  return gpu_cache.amd_available;
#endif

#ifdef __linux__
  // Check for AMD GPU via lspci or /sys
  std::ifstream lspci_proc("/proc/bus/pci/devices");
  if (lspci_proc.is_open()) {
    std::string line;
    while (std::getline(lspci_proc, line)) {
      if (line.find("1002") != std::string::npos) {  // AMD vendor ID
        return true;
      }
    }
  }

  // Check ROCm installation
  std::ifstream rocm_file("/opt/rocm/bin/rocm-smi");
  if (rocm_file.good()) {
    return true;
  }
#endif

#ifdef _WIN32
  // Windows: Check via WMI or device manager
  // TODO: Implement Windows AMD GPU detection
#endif

  return false;
}

std::string detectAMDGPUName() {
#ifdef __APPLE__
  // Get AMD GPU name from system_profiler on Mac
  FILE* pipe = popen(
      "system_profiler SPDisplaysDataType | grep -i 'Chipset Model' | grep -i 'AMD\\|Radeon' | head -1 2>/dev/null",
      "r");
  if (pipe) {
    char buffer[256];
    std::string result;
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
      result += buffer;
    }
    pclose(pipe);

    if (!result.empty()) {
      // Extract GPU name from system_profiler output
      size_t pos = result.find(":");
      if (pos != std::string::npos) {
        std::string gpu_name = result.substr(pos + 1);
        // Remove whitespace and newlines
        gpu_name.erase(0, gpu_name.find_first_not_of(" \t\n\r"));
        gpu_name.erase(gpu_name.find_last_not_of(" \t\n\r") + 1);
        return gpu_name;
      }
    }
  }
#endif

#ifdef __linux__
  // Try to get GPU name from ROCm
  FILE* pipe = popen("rocm-smi --showid 2>/dev/null", "r");
  if (pipe) {
    char buffer[256];
    std::string result;
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
      result += buffer;
    }
    pclose(pipe);

    if (!result.empty() && result.find("GPU") != std::string::npos) {
      return "AMD ROCm GPU";
    }
  }

  // Fallback to generic name
  return "AMD GPU";
#endif
  return "AMD GPU";
}

bool checkIntelGPU() {
  gpu_cache.initialize();
#ifdef __APPLE__
  return gpu_cache.intel_available;
#endif

#ifdef __linux__
  // Check for Intel GPU via /sys
  std::ifstream lspci_proc("/proc/bus/pci/devices");
  if (lspci_proc.is_open()) {
    std::string line;
    while (std::getline(lspci_proc, line)) {
      if (line.find("8086") != std::string::npos) {  // Intel vendor ID
        return true;
      }
    }
  }

  // Check for Intel oneAPI/OpenCL
  std::ifstream oneapi_file(
      "/opt/intel/oneapi/compiler/latest/linux/bin/intel64/icc");
  if (oneapi_file.good()) {
    return true;
  }
#endif

#ifdef _WIN32
  // Windows: Check for Intel GPU
  // TODO: Implement Windows Intel GPU detection
#endif

  return false;
}

std::string detectIntelGPUName() {
#ifdef __APPLE__
  // Get Intel GPU name from system_profiler on Mac
  FILE* pipe = popen(
      "system_profiler SPDisplaysDataType | grep -i 'Chipset Model' | grep -i 'Intel' | head -1 2>/dev/null",
      "r");
  if (pipe) {
    char buffer[256];
    std::string result;
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
      result += buffer;
    }
    pclose(pipe);

    if (!result.empty()) {
      // Extract GPU name from system_profiler output
      size_t pos = result.find(":");
      if (pos != std::string::npos) {
        std::string gpu_name = result.substr(pos + 1);
        // Remove whitespace and newlines
        gpu_name.erase(0, gpu_name.find_first_not_of(" \t\n\r"));
        gpu_name.erase(gpu_name.find_last_not_of(" \t\n\r") + 1);
        return gpu_name;
      }
    }
  }
#endif

#ifdef __linux__
  // Try to detect Intel GPU model
  FILE* pipe = popen("lspci | grep -i intel | grep -i vga 2>/dev/null", "r");
  if (pipe) {
    char buffer[256];
    std::string result;
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
      result += buffer;
    }
    pclose(pipe);

    if (!result.empty()) {
      return "Intel GPU";
    }
  }
#endif
  return "Intel GPU";
}

#ifdef __APPLE__
bool checkAppleGPU() {
  // Check for Apple Silicon only (not Intel Macs with discrete GPUs)
  size_t size = 0;
  if (sysctlbyname("hw.optional.arm64", nullptr, &size, nullptr, 0) == 0) {
    uint32_t val = 0;
    if (sysctlbyname("hw.optional.arm64", &val, &size, nullptr, 0) == 0) {
      return val == 1;  // Only Apple Silicon has integrated Apple GPU
    }
  }

  return false;  // Intel Macs don't have Apple GPUs
}

std::string detectAppleGPUName() {
  // Detect Apple Silicon GPU model
  size_t size = 0;
  char model[64];
  if (sysctlbyname("hw.model", nullptr, &size, nullptr, 0) == 0) {
    if (size < sizeof(model) &&
        sysctlbyname("hw.model", model, &size, nullptr, 0) == 0) {
      std::string model_str(model);

      // Map Apple Silicon models to GPU names
      if (model_str.find("MacBookAir10") != std::string::npos ||
          model_str.find("Macmini9") != std::string::npos ||
          model_str.find("MacBookPro17") != std::string::npos) {
        return "Apple M1 GPU";
      } else if (model_str.find("MacBookAir") != std::string::npos ||
                 model_str.find("MacBookPro18") != std::string::npos ||
                 model_str.find("Macmini") != std::string::npos) {
        return "Apple M1 Pro/Max GPU";
      } else if (model_str.find("Mac13") != std::string::npos) {
        return "Apple M2 GPU";
      } else if (model_str.find("Mac14") != std::string::npos ||
                 model_str.find("Mac15") != std::string::npos) {
        return "Apple M3 GPU";
      }
    }
  }

  return "Apple Silicon GPU";
}
#else
bool checkAppleGPU() {
  return false;  // Apple GPU not available on non-Apple platforms
}

std::string detectAppleGPUName() {
  return "Apple GPU";  // Not available on non-Apple platforms
}
#endif

int detectNVIDIAGPUMemoryMB() {
#ifdef __APPLE__
  // Get NVIDIA GPU VRAM from system_profiler on Mac
  FILE* pipe = popen(
      "system_profiler SPDisplaysDataType | grep -A 5 -i 'nvidia\\|geforce\\|quadro' | grep -i 'VRAM\\|Total.*GB' 2>/dev/null",
      "r");
  if (pipe) {
    char buffer[256];
    std::string result;
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
      result += buffer;
    }
    pclose(pipe);

    if (!result.empty()) {
      // Parse VRAM size from system_profiler output
      size_t gb_pos = result.find("GB");
      if (gb_pos != std::string::npos) {
        // Find number before "GB"
        size_t start = gb_pos;
        while (start > 0 &&
               (result[start - 1] == ' ' || result[start - 1] == ':')) {
          start--;
        }
        while (start > 0 && std::isdigit(result[start - 1])) {
          start--;
        }
        if (start < gb_pos) {
          std::string gb_str = result.substr(start, gb_pos - start);
          try {
            int gb = std::stoi(gb_str);
            return gb * 1024;  // Convert GB to MB
          } catch (const std::exception&) {
            // Ignore parsing errors, continue to return 0
          }
        }
      }
    }
  }
#endif

#ifdef __linux__
#ifdef CI
  // In CI environment, avoid external command execution
  return 0;
#else
  // Try to get VRAM from nvidia-smi
  FILE* pipe = popen(
      "nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null",
      "r");
  if (pipe) {
    char buffer[256];
    std::string result;
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
      result += buffer;
    }
    pclose(pipe);

    if (!result.empty()) {
      // nvidia-smi reports memory in MB
      try {
        int memory_mb = std::stoi(result);
        return memory_mb;
      } catch (const std::exception&) {
        // Ignore parsing errors
      }
    }
  }
#endif
#endif
  return 0;  // Unknown or not available
}

int detectAMDGPUMemoryMB() {
#ifdef __APPLE__
  // Get AMD GPU VRAM from system_profiler on Mac
  FILE* pipe = popen(
      "system_profiler SPDisplaysDataType | grep -A 5 -i 'AMD\\|Radeon' | grep -i 'VRAM\\|Total.*GB' 2>/dev/null",
      "r");
  if (pipe) {
    char buffer[256];
    std::string result;
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
      result += buffer;
    }
    pclose(pipe);

    if (!result.empty()) {
      // Parse VRAM size from system_profiler output
      size_t gb_pos = result.find("GB");
      if (gb_pos != std::string::npos) {
        // Find number before "GB"
        size_t start = gb_pos;
        while (start > 0 &&
               (result[start - 1] == ' ' || result[start - 1] == ':')) {
          start--;
        }
        while (start > 0 && std::isdigit(result[start - 1])) {
          start--;
        }
        if (start < gb_pos) {
          std::string gb_str = result.substr(start, gb_pos - start);
          try {
            int gb = std::stoi(gb_str);
            return gb * 1024;  // Convert GB to MB
          } catch (const std::exception&) {
            // Ignore parsing errors, continue to return 0
          }
        }
      }
    }
  }
#endif

#ifdef __linux__
#ifdef CI
  // In CI environment, avoid external command execution
  return 0;
#else
  // Try to get VRAM from ROCm
  FILE* pipe = popen("rocm-smi --showmeminfo vram --csv 2>/dev/null", "r");
  if (pipe) {
    char buffer[256];
    std::string result;
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
      result += buffer;
    }
    pclose(pipe);

    if (!result.empty() && result.find(",") != std::string::npos) {
      // Parse ROCm memory info
      size_t pos = result.find_last_of(",");
      if (pos != std::string::npos) {
        std::string mem_str = result.substr(pos + 1);
        // ROCm reports in bytes, convert to MB
        try {
          long long bytes = std::stoll(mem_str);
          return static_cast<int>(bytes / (1024 * 1024));
        } catch (const std::exception&) {
          // Ignore parsing errors
        }
      }
    }
  }
#endif
#endif
  return 0;  // Unknown or not available
}

int detectIntelGPUMemoryMB() {
  // Intel integrated GPUs typically use shared system memory
  // We can report a portion of system RAM as available to Intel GPU
#ifdef __APPLE__
  size_t size = sizeof(uint64_t);
  uint64_t system_memory = 0;
  if (sysctlbyname("hw.memsize", &system_memory, &size, nullptr, 0) == 0) {
    // Intel GPU can use about 1/4 to 1/2 of system memory
    // Report 1/4 as conservative estimate
    return static_cast<int>((system_memory / 4) /
                            (1024 * 1024));  // Convert to MB
  }
#endif

#ifdef __linux__
  // Get system memory from /proc/meminfo
  std::ifstream meminfo("/proc/meminfo");
  if (meminfo.is_open()) {
    std::string line;
    while (std::getline(meminfo, line)) {
      if (line.find("MemTotal:") != std::string::npos) {
        size_t pos = line.find_first_of("0123456789");
        if (pos != std::string::npos) {
          size_t end = line.find(" kB");
          if (end != std::string::npos) {
            std::string mem_kb = line.substr(pos, end - pos);
            try {
              int kb = std::stoi(mem_kb);
              // Report 1/4 of system memory for Intel integrated GPU
              return (kb / 4) / 1024;  // Convert KB to MB
            } catch (const std::exception&) {
              // Ignore parsing errors
            }
          }
        }
        break;
      }
    }
  }
#endif
  return 0;  // Unknown or not available
}

#ifdef __APPLE__
int detectAppleGPUMemoryMB() {
  // Apple Silicon uses unified memory architecture
  // The GPU shares system memory with the CPU
  size_t size = sizeof(uint64_t);
  uint64_t system_memory = 0;
  if (sysctlbyname("hw.memsize", &system_memory, &size, nullptr, 0) == 0) {
    // On Apple Silicon, GPU can access all system memory
    // But we'll report a reasonable portion for GPU workloads
    // Typical allocation is about 60-75% of system memory for GPU
    uint64_t gpu_memory = (system_memory * 3) / 4;  // 75% of system memory
    return static_cast<int>(gpu_memory / (1024 * 1024));  // Convert to MB
  }
  return 0;  // Unknown or not available
}
#else
int detectAppleGPUMemoryMB() {
  // Apple GPU not available on non-Apple platforms
  return 0;
}
#endif

}  // anonymous namespace

}  // namespace MLLib
