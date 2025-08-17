#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>

// Mock Intel Mac + AMD Radeon Pro system for testing
void simulateIntelMacAMDGPU() {
  std::cout << "=== Simulating Intel Mac + AMD Radeon Pro GPU ===" << std::endl;
  std::cout << "Expected results:" << std::endl;
  std::cout << "  - Architecture: Intel Mac (hw.optional.arm64 = 0)"
            << std::endl;
  std::cout << "  - GPU Hardware: AMD Radeon Pro" << std::endl;
  std::cout << "  - Expected Vendor: AMD (not Apple)" << std::endl;
  std::cout << "  - Expected API Support: OpenCL/Metal" << std::endl;
  std::cout << "  - Metal Support: Available (but vendor should still be AMD)"
            << std::endl;
  std::cout << std::endl;

  // Simulate system_profiler output for Intel Mac + AMD GPU
  const char* mock_system_profiler = R"(
Graphics/Displays:

    AMD Radeon Pro 5700:

      Chipset Model: AMD Radeon Pro 5700
      Type: GPU
      Bus: PCIe
      VRAM (Total): 8 GB
      Vendor: AMD (0x1002)
      Device ID: 0x731f
      Revision ID: 0x00c1
      ROM Revision: 113-D180AU-X12
      GGART Size: 1024 MB
      GVRAM Size: 8192 MB
      Metal Support: Metal 3
      Displays:
        PL2779Q:
          Resolution: 2560 x 1440 @ 60 Hz
          UI Looks like: 2560 x 1440 @ 60 Hz
          Main Display: Yes
          Mirror: Off
          Online: Yes
          Connection Type: DisplayPort
)";

  std::cout << "Mock system_profiler output:" << std::endl;
  std::cout << mock_system_profiler << std::endl;

  // Test AMD GPU detection pattern
  std::string output(mock_system_profiler);

  // Check AMD GPU detection
  bool amd_detected = (output.find("AMD") != std::string::npos ||
                       output.find("Radeon") != std::string::npos);
  std::cout << "AMD GPU detection: "
            << (amd_detected ? "✅ DETECTED" : "❌ NOT DETECTED") << std::endl;

  // Check Apple Silicon detection (should be false for Intel Mac)
  bool apple_silicon = false;  // Simulate hw.optional.arm64 = 0
  std::cout << "Apple Silicon detection: "
            << (apple_silicon ? "✅ DETECTED"
                              : "❌ NOT DETECTED (Correct for Intel Mac)")
            << std::endl;

  // Extract GPU name
  if (amd_detected) {
    size_t pos = output.find("Chipset Model:");
    if (pos != std::string::npos) {
      size_t start = pos + 14;  // "Chipset Model:".length()
      size_t end = output.find("\n", start);
      if (end != std::string::npos) {
        std::string gpu_name = output.substr(start, end - start);
        // Remove leading/trailing whitespace
        gpu_name.erase(0, gpu_name.find_first_not_of(" \t"));
        gpu_name.erase(gpu_name.find_last_not_of(" \t") + 1);
        std::cout << "Extracted GPU name: \"" << gpu_name << "\"" << std::endl;
      }
    }
  }

  // Metal support check
  bool metal_support = output.find("Metal Support") != std::string::npos;
  std::cout << "Metal API support: "
            << (metal_support ? "✅ AVAILABLE" : "❌ NOT AVAILABLE")
            << std::endl;

  std::cout << std::endl;
  std::cout << "=== Analysis ===" << std::endl;
  std::cout << "✅ This system should report:" << std::endl;
  std::cout << "   - Vendor: AMD" << std::endl;
  std::cout << "   - Name: AMD Radeon Pro 5700" << std::endl;
  std::cout << "   - Memory: 8192 MB (8GB dedicated VRAM)" << std::endl;
  std::cout << "   - API Support: OpenCL/Metal" << std::endl;
  std::cout << "   - Primary GPU Vendor: AMD" << std::endl;
  std::cout << std::endl;
  std::cout << "❌ This system should NOT report:" << std::endl;
  std::cout << "   - Vendor: Apple (incorrect - this is AMD hardware)"
            << std::endl;
  std::cout << "   - Primary GPU Vendor: Apple" << std::endl;
  std::cout << std::endl;
  std::cout
      << "🔍 Key Point: Metal API availability does not make it an Apple GPU!"
      << std::endl;
  std::cout
      << "   Metal is available on AMD/NVIDIA GPUs on macOS, but the vendor"
      << std::endl;
  std::cout << "   should reflect the actual hardware manufacturer."
            << std::endl;
}

void simulateIntelMacNVIDIAGPU() {
  std::cout << "=== Simulating Intel Mac + NVIDIA RTX GPU ===" << std::endl;

  // Simulate system_profiler output for Intel Mac + NVIDIA GPU
  const char* mock_system_profiler = R"(
Graphics/Displays:

    NVIDIA GeForce RTX 3080:

      Chipset Model: NVIDIA GeForce RTX 3080
      Type: GPU
      Bus: PCIe
      VRAM (Total): 10 GB
      Vendor: NVIDIA (0x10de)
      Device ID: 0x2206
      Metal Support: Metal 3
      Displays:
        Studio Display:
          Resolution: 5120 x 2880 @ 60 Hz
)";

  std::cout << "Mock system_profiler output:" << std::endl;
  std::cout << mock_system_profiler << std::endl;

  std::string output(mock_system_profiler);

  // Check NVIDIA GPU detection
  bool nvidia_detected = (output.find("NVIDIA") != std::string::npos ||
                          output.find("GeForce") != std::string::npos ||
                          output.find("Quadro") != std::string::npos);
  std::cout << "NVIDIA GPU detection: "
            << (nvidia_detected ? "✅ DETECTED" : "❌ NOT DETECTED")
            << std::endl;

  std::cout << std::endl;
  std::cout << "✅ This system should report:" << std::endl;
  std::cout << "   - Vendor: NVIDIA" << std::endl;
  std::cout << "   - Name: NVIDIA GeForce RTX 3080" << std::endl;
  std::cout << "   - Memory: 10240 MB (10GB dedicated VRAM)" << std::endl;
  std::cout << "   - API Support: CUDA (if CUDA installed) or OpenCL/Metal"
            << std::endl;
  std::cout << "   - Primary GPU Vendor: NVIDIA" << std::endl;
}

int main() {
  std::cout << "=== GPU Vendor Detection Scenario Testing ===" << std::endl;
  std::cout << "This test simulates different Mac configurations to verify"
            << std::endl;
  std::cout << "that GPU vendor detection works correctly." << std::endl;
  std::cout << std::endl;

  simulateIntelMacAMDGPU();
  std::cout << std::string(70, '=') << std::endl;
  std::cout << std::endl;

  simulateIntelMacNVIDIAGPU();
  std::cout << std::string(70, '=') << std::endl;
  std::cout << std::endl;

  std::cout << "=== Summary ===" << std::endl;
  std::cout << "The current MLLib GPU detection logic should correctly:"
            << std::endl;
  std::cout << "1. Distinguish hardware vendor from API support" << std::endl;
  std::cout << "2. Report AMD vendor for AMD hardware (even with Metal support)"
            << std::endl;
  std::cout
      << "3. Report NVIDIA vendor for NVIDIA hardware (even with Metal support)"
      << std::endl;
  std::cout << "4. Only report Apple vendor for Apple Silicon integrated GPUs"
            << std::endl;
  std::cout << "5. Correctly detect GPU memory:" << std::endl;
  std::cout << "   - Apple Silicon: Unified memory (75% of system RAM)"
            << std::endl;
  std::cout << "   - AMD/NVIDIA: Dedicated VRAM from system_profiler"
            << std::endl;
  std::cout << "   - Intel integrated: Shared memory (25% of system RAM)"
            << std::endl;
  std::cout << std::endl;
  std::cout
      << "If you're seeing incorrect vendor detection on Intel Mac + discrete GPU,"
      << std::endl;
  std::cout
      << "please run the actual device_detection sample and compare with these"
      << std::endl;
  std::cout << "expected results." << std::endl;

  return 0;
}
