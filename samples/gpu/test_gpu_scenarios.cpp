#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// Simulate different system configurations for testing GPU detection
struct MockSystemInfo {
  std::string name;
  std::string system_profiler_output;
  bool is_apple_silicon;
  bool has_cuda;
  bool has_rocm;
  std::string expected_primary_vendor;
  std::string expected_api_support;
};

// Mock different system configurations
std::vector<MockSystemInfo> test_systems = {
    {"Apple M1 MacBook Pro",
     "Graphics/Displays:\n\n    Apple M1:\n\n      Chipset Model: Apple M1\n      Type: GPU\n      Bus: Built-In\n",
     true, false, false, "Apple", "Metal"},
    {"Intel Mac + AMD Radeon Pro 5700",
     "Graphics/Displays:\n\n    AMD Radeon Pro 5700:\n\n      Chipset Model: AMD Radeon Pro 5700\n      Type: GPU\n      Bus: PCIe\n      VRAM (Total): 8 GB\n",
     false, false, false, "AMD", "Metal"},
    {"Intel Mac + NVIDIA RTX 3080",
     "Graphics/Displays:\n\n    NVIDIA GeForce RTX 3080:\n\n      Chipset Model: NVIDIA GeForce RTX 3080\n      Type: GPU\n      Bus: PCIe\n      VRAM (Total): 10 GB\n",
     false, true, false, "NVIDIA", "CUDA"},
    {"Intel Mac + Intel Iris Plus Graphics",
     "Graphics/Displays:\n\n    Intel Iris Plus Graphics:\n\n      Chipset Model: Intel Iris Plus Graphics\n      Type: GPU\n      Bus: Built-In\n",
     false, false, false, "Intel", "Metal/OpenCL"}};

enum class GPUVendor {
  UNKNOWN = 0,
  NVIDIA = 1,
  AMD = 2,
  INTEL_GPU = 3,
  APPLE = 4
};

struct MockGPUInfo {
  GPUVendor vendor;
  std::string name;
  int memory_mb;
  bool compute_capable;
  std::string api_support;
};

// Mock detection functions based on system_profiler output
bool mockCheckNVIDIAGPU(const std::string& output) {
  return output.find("NVIDIA") != std::string::npos ||
      output.find("GeForce") != std::string::npos ||
      output.find("Quadro") != std::string::npos;
}

bool mockCheckAMDGPU(const std::string& output) {
  return output.find("AMD") != std::string::npos ||
      output.find("Radeon") != std::string::npos;
}

bool mockCheckIntelGPU(const std::string& output) {
  return output.find("Intel") != std::string::npos &&
      (output.find("Graphics") != std::string::npos ||
       output.find("Iris") != std::string::npos ||
       output.find("UHD") != std::string::npos);
}

bool mockCheckAppleGPU(bool is_apple_silicon) {
  return is_apple_silicon;
}

std::string mockDetectGPUName(GPUVendor vendor, const std::string& output) {
  switch (vendor) {
  case GPUVendor::NVIDIA: {
    size_t model_pos = output.find("Chipset Model:");
    if (model_pos != std::string::npos) {
      size_t start = model_pos + 14;  // "Chipset Model:".length()
      size_t end = output.find("\n", start);
      if (end != std::string::npos) {
        std::string name = output.substr(start, end - start);
        // Remove leading/trailing whitespace
        name.erase(0, name.find_first_not_of(" \t"));
        name.erase(name.find_last_not_of(" \t") + 1);
        return name;
      }
    }
    return "NVIDIA GPU";
  }
  case GPUVendor::AMD: {
    size_t model_pos = output.find("Chipset Model:");
    if (model_pos != std::string::npos) {
      size_t start = model_pos + 14;
      size_t end = output.find("\n", start);
      if (end != std::string::npos) {
        std::string name = output.substr(start, end - start);
        name.erase(0, name.find_first_not_of(" \t"));
        name.erase(name.find_last_not_of(" \t") + 1);
        return name;
      }
    }
    return "AMD GPU";
  }
  case GPUVendor::INTEL_GPU: {
    size_t model_pos = output.find("Chipset Model:");
    if (model_pos != std::string::npos) {
      size_t start = model_pos + 14;
      size_t end = output.find("\n", start);
      if (end != std::string::npos) {
        std::string name = output.substr(start, end - start);
        name.erase(0, name.find_first_not_of(" \t"));
        name.erase(name.find_last_not_of(" \t") + 1);
        return name;
      }
    }
    return "Intel GPU";
  }
  case GPUVendor::APPLE: return "Apple M1 GPU";
  default: return "Unknown GPU";
  }
}

std::vector<MockGPUInfo> mockDetectGPUs(const MockSystemInfo& system) {
  std::vector<MockGPUInfo> gpus;

  // 1. Check NVIDIA GPUs first
  if (mockCheckNVIDIAGPU(system.system_profiler_output)) {
    MockGPUInfo nvidia_gpu;
    nvidia_gpu.vendor = GPUVendor::NVIDIA;
    nvidia_gpu.name =
        mockDetectGPUName(GPUVendor::NVIDIA, system.system_profiler_output);
    nvidia_gpu.memory_mb = 10240;  // Mock 10GB
    nvidia_gpu.compute_capable = system.has_cuda;
    nvidia_gpu.api_support = system.has_cuda ? "CUDA" : "Metal/OpenCL";
    gpus.push_back(nvidia_gpu);
  }

  // 2. Check AMD GPUs
  if (mockCheckAMDGPU(system.system_profiler_output)) {
    MockGPUInfo amd_gpu;
    amd_gpu.vendor = GPUVendor::AMD;
    amd_gpu.name =
        mockDetectGPUName(GPUVendor::AMD, system.system_profiler_output);
    amd_gpu.memory_mb = 8192;  // Mock 8GB
    amd_gpu.compute_capable = true;
    amd_gpu.api_support = system.has_rocm ? "ROCm" : "Metal/OpenCL";
    gpus.push_back(amd_gpu);
  }

  // 3. Check Intel GPUs
  if (mockCheckIntelGPU(system.system_profiler_output)) {
    MockGPUInfo intel_gpu;
    intel_gpu.vendor = GPUVendor::INTEL_GPU;
    intel_gpu.name =
        mockDetectGPUName(GPUVendor::INTEL_GPU, system.system_profiler_output);
    intel_gpu.memory_mb = 0;  // Shared memory
    intel_gpu.compute_capable = true;
    intel_gpu.api_support = "Metal/OpenCL";
    gpus.push_back(intel_gpu);
  }

  // 4. Check Apple Silicon GPUs (only on Apple Silicon)
  if (mockCheckAppleGPU(system.is_apple_silicon)) {
    MockGPUInfo apple_gpu;
    apple_gpu.vendor = GPUVendor::APPLE;
    apple_gpu.name =
        mockDetectGPUName(GPUVendor::APPLE, system.system_profiler_output);
    apple_gpu.memory_mb = 0;  // Unified memory
    apple_gpu.compute_capable = true;
    apple_gpu.api_support = "Metal";
    gpus.push_back(apple_gpu);
  }

  return gpus;
}

GPUVendor mockGetPrimaryGPUVendor(const std::vector<MockGPUInfo>& gpus) {
  if (gpus.empty()) {
    return GPUVendor::UNKNOWN;
  }

  // Priority order: NVIDIA > AMD > Apple > Intel
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

std::string vendorToString(GPUVendor vendor) {
  switch (vendor) {
  case GPUVendor::NVIDIA: return "NVIDIA";
  case GPUVendor::AMD: return "AMD";
  case GPUVendor::INTEL_GPU: return "Intel";
  case GPUVendor::APPLE: return "Apple";
  default: return "Unknown";
  }
}

int main() {
  std::cout << "=== GPU Detection Scenario Testing ===\n\n";

  for (const auto& system : test_systems) {
    std::cout << "Testing: " << system.name << "\n";
    std::cout << "System Profile Output:\n"
              << system.system_profiler_output << "\n";

    auto gpus = mockDetectGPUs(system);
    auto primary_vendor = mockGetPrimaryGPUVendor(gpus);

    std::cout << "Detected " << gpus.size() << " GPU(s):\n";
    for (size_t i = 0; i < gpus.size(); ++i) {
      const auto& gpu = gpus[i];
      std::cout << "  GPU " << (i + 1) << ":\n";
      std::cout << "    Vendor: " << vendorToString(gpu.vendor) << "\n";
      std::cout << "    Name: " << gpu.name << "\n";
      std::cout << "    Memory: " << gpu.memory_mb << " MB\n";
      std::cout << "    API Support: " << gpu.api_support << "\n";
    }

    std::cout << "Primary GPU Vendor: " << vendorToString(primary_vendor)
              << "\n";
    std::cout << "Expected: " << system.expected_primary_vendor << "\n";

    bool correct =
        vendorToString(primary_vendor) == system.expected_primary_vendor;
    std::cout << "Result: " << (correct ? "✅ CORRECT" : "❌ INCORRECT")
              << "\n";
    std::cout << "\n" << std::string(60, '=') << "\n\n";
  }

  return 0;
}
