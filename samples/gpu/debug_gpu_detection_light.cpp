/**
 * @file debug_gpu_detection.cpp
 * @brief Debug GPU detection logic
 */

#include <cstdio>
#include <cstdlib>
#include <string>

// macOS specific includes
#ifdef __APPLE__
#include <sys/sysctl.h>
#endif

int main() {
  printf("=== GPU Detection Debug ===\n");

#ifdef __APPLE__
  // Check if this is Apple Silicon
  size_t size = 0;
  printf("Checking for Apple Silicon...\n");
  if (sysctlbyname("hw.optional.arm64", nullptr, &size, nullptr, 0) == 0) {
    uint32_t val = 0;
    if (sysctlbyname("hw.optional.arm64", &val, &size, nullptr, 0) == 0) {
      printf("hw.optional.arm64 = %u\n", val);
      if (val == 1) {
        printf("-> This is Apple Silicon (ARM64)\n");
      } else {
        printf("-> This is Intel Mac (x86_64)\n");
      }
    }
  } else {
    printf("-> Cannot determine CPU architecture\n");
  }

  // Check system_profiler output
  printf("\nChecking system_profiler output...\n");
  FILE* pipe = popen(
      "system_profiler SPDisplaysDataType | grep -i 'Chipset Model' 2>/dev/null",
      "r");
  if (pipe) {
    char buffer[256];
    printf("All GPUs found:\n");
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
      printf("  %s", buffer);
    }
    pclose(pipe);
  }

  // Check for AMD specifically
  printf("\nChecking for AMD GPUs...\n");
  pipe = popen(
      "system_profiler SPDisplaysDataType | grep -i 'Chipset Model' | grep -i 'AMD\\|Radeon' 2>/dev/null",
      "r");
  if (pipe) {
    char buffer[256];
    std::string result;
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
      result += buffer;
    }
    pclose(pipe);

    if (!result.empty()) {
      printf("AMD GPU found: %s", result.c_str());
    } else {
      printf("No AMD GPU detected\n");
    }
  }

  // Check for Intel specifically
  printf("\nChecking for Intel GPUs...\n");
  pipe = popen(
      "system_profiler SPDisplaysDataType | grep -i 'Chipset Model' | grep -i 'Intel' 2>/dev/null",
      "r");
  if (pipe) {
    char buffer[256];
    std::string result;
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
      result += buffer;
    }
    pclose(pipe);

    if (!result.empty()) {
      printf("Intel GPU found: %s", result.c_str());
    } else {
      printf("No Intel GPU detected\n");
    }
  }
#else
  // Non-Apple platforms
  printf("macOS-specific GPU detection not available on this platform.\n");
  printf("This debug tool is designed for macOS systems.\n");
  
  // On Linux, you could use lspci instead
  printf("\nTrying Linux GPU detection...\n");
  FILE* pipe = popen("lspci | grep -i 'vga\\|3d\\|display' 2>/dev/null", "r");
  if (pipe) {
    char buffer[256];
    printf("GPUs found via lspci:\n");
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
      printf("  %s", buffer);
    }
    pclose(pipe);
  } else {
    printf("Could not execute lspci command\n");
  }
#endif

  return 0;
}
