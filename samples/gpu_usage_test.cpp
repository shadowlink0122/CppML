/**
 * @file gpu_usage_test.cpp
 * @brief Test actual GPU computation usage vs CPU fallback with large matrices
 * 
 * This sample demonstrates:
 * - Whether GPU computation is actually being used
 * - Performance difference between CPU and GPU across different matrix sizes
 * - GPU effectiveness for large-scale computations
 */

#include <MLLib.hpp>
#include <cstdio>
#include <chrono>
#include <vector>
#include <string>
#include <cmath>

using namespace MLLib;

void testMatrixMultiplication() {
  printf("=== MLLib GPU Performance Benchmark ===\n");
  printf("Testing GPU effectiveness across different matrix sizes\n\n");
  
  // Test with multiple matrix sizes to show GPU effectiveness
  std::vector<size_t> sizes = {256, 512, 1024, 2048};
  
  for (size_t size : sizes) {
    printf("🔸 Testing %zdx%zd matrix multiplication 🔸\n", size, size);
    printf("%s\n", std::string(50, '-').c_str());
    
    // Create test matrices
    NDArray a({size, size});
    NDArray b({size, size});
    NDArray result_cpu({size, size});
    NDArray result_gpu({size, size});
    
    // Initialize with test data
    printf("🔄 Initializing matrices with test data...\n");
    for (size_t i = 0; i < size * size; i++) {
      a.data()[i] = static_cast<double>(i % 100) / 100.0;
      b.data()[i] = static_cast<double>((i * 2) % 100) / 100.0;
    }
    
    // Test CPU computation
    printf("\n🖥️  CPU Computation:\n");
    Device::setDevice(DeviceType::CPU);
    
    auto start_cpu = std::chrono::high_resolution_clock::now();
    
    try {
      Backend::Backend::matmul(a, b, result_cpu);
      auto end_cpu = std::chrono::high_resolution_clock::now();
      auto duration_cpu = std::chrono::duration_cast<std::chrono::microseconds>(end_cpu - start_cpu);
      printf("⏱️  CPU time: %lld μs (%.2f ms)\n", duration_cpu.count(), duration_cpu.count() / 1000.0);
      printf("✅ CPU computation: SUCCESS\n");
      
      // Test GPU computation
      printf("\n🚀 GPU Computation:\n");
      Device::setDevice(DeviceType::GPU);
      
      auto start_gpu = std::chrono::high_resolution_clock::now();
      
      try {
        Backend::Backend::matmul(a, b, result_gpu);
        auto end_gpu = std::chrono::high_resolution_clock::now();
        auto duration_gpu = std::chrono::duration_cast<std::chrono::microseconds>(end_gpu - start_gpu);
        printf("⏱️  GPU time: %lld μs (%.2f ms)\n", duration_gpu.count(), duration_gpu.count() / 1000.0);
        printf("✅ GPU computation: SUCCESS\n");
        
        // Calculate speedup
        double speedup = static_cast<double>(duration_cpu.count()) / static_cast<double>(duration_gpu.count());
        printf("\n📊 Performance Analysis:\n");
        if (speedup > 1.2) {
          printf("🎉 GPU is %.2fx faster than CPU! (Excellent acceleration)\n", speedup);
        } else if (speedup > 1.05) {
          printf("👍 GPU is %.2fx faster than CPU (Good acceleration)\n", speedup);
        } else if (speedup > 0.95) {
          printf("📏 GPU and CPU performance are similar (%.2fx)\n", speedup);
        } else {
          printf("🐌 CPU is %.2fx faster than GPU (overhead dominates for this size)\n", 1.0/speedup);
        }
        
        // Performance per element analysis
        double cpu_ns_per_element = (duration_cpu.count() * 1000.0) / (size * size * size);
        double gpu_ns_per_element = (duration_gpu.count() * 1000.0) / (size * size * size);
        printf("🔬 CPU: %.2f ns/element, GPU: %.2f ns/element\n", cpu_ns_per_element, gpu_ns_per_element);
        
        // Verify correctness
        double max_diff = 0.0;
        for (size_t i = 0; i < size * size; i++) {
          double diff = std::abs(result_cpu.data()[i] - result_gpu.data()[i]);
          max_diff = std::max(max_diff, diff);
        }
        
        printf("\n🔍 Numerical Verification:\n");
        printf("Max difference: %.10f\n", max_diff);
        
        // Metal uses float precision, so we expect some difference from CPU double precision
        if (max_diff < 1e-6) {
          printf("✅ Excellent precision match (Metal float vs CPU double)\n");
        } else if (max_diff < 1e-3) {
          printf("✅ Good precision (expected for Metal float vs CPU double)\n");
        } else if (max_diff < 1e-1) {
          printf("⚠️  Acceptable precision difference (Metal float limitation)\n");
        } else {
          printf("❌ Large precision difference - potential algorithm issue\n");
        }
        
      } catch (const std::exception& e) {
        printf("❌ GPU computation failed: %s\n", e.what());
        printf("🔄 Likely falling back to CPU computation\n");
      }
      
    } catch (const std::exception& e) {
      printf("❌ CPU computation failed: %s\n", e.what());
      continue;
    }
    
    printf("\n%s\n\n", std::string(60, '=').c_str());
  }
  
  printf("💡 Interpretation:\n");
  printf("- Larger matrices (1024+) should show GPU advantages\n");  
  printf("- Small matrices may show CPU advantages due to GPU overhead\n");
  printf("- Metal Performance Shaders provide maximum optimization for Apple Silicon\n");
  printf("- Metal uses float32, CPU uses double64 - small precision differences expected\n");
}

void testBackendSelection() {
  printf("=== GPU Backend Selection Test ===\n");
  
  if (!Device::isGPUAvailable()) {
    printf("❌ No GPU available for testing\n");
    return;
  }
  
  auto backends = Backend::Backend::getAvailableGPUBackends();
  printf("Available GPU backends: %zu\n", backends.size());
  
  for (auto backend : backends) {
    switch (backend) {
      case MLLib::Backend::GPUBackendType::CUDA:
        printf("  - CUDA\n");
        break;
      case MLLib::Backend::GPUBackendType::ROCM:
        printf("  - ROCm\n");
        break;
      case MLLib::Backend::GPUBackendType::METAL:
        printf("  - Metal\n");
        break;
      case MLLib::Backend::GPUBackendType::ONEAPI:
        printf("  - oneAPI\n");
        break;
      default:
        break;
    }
  }
  
  // Test backend switching for available backends
  printf("\nTesting backend switching...\n");
  for (auto backend : backends) {
    std::string backend_name;
    switch (backend) {
      case MLLib::Backend::GPUBackendType::CUDA: backend_name = "CUDA"; break;
      case MLLib::Backend::GPUBackendType::ROCM: backend_name = "ROCm"; break; 
      case MLLib::Backend::GPUBackendType::METAL: backend_name = "Metal"; break;
      case MLLib::Backend::GPUBackendType::ONEAPI: backend_name = "oneAPI"; break;
      default: continue;
    }
    
    try {
      Backend::Backend::setPreferredGPUBackend(backend);
      auto current = Backend::Backend::getCurrentGPUBackend();
      std::string current_name;
      switch (current) {
        case MLLib::Backend::GPUBackendType::CUDA: current_name = "CUDA"; break;
        case MLLib::Backend::GPUBackendType::ROCM: current_name = "ROCm"; break;
        case MLLib::Backend::GPUBackendType::METAL: current_name = "Metal"; break; 
        case MLLib::Backend::GPUBackendType::ONEAPI: current_name = "oneAPI"; break;
        default: current_name = "Unknown"; break;
      }
      printf("  Set %s: SUCCESS (current: %s)\n", backend_name.c_str(), current_name.c_str());
    } catch (const std::exception& e) {
      printf("  Set %s: FAILED (%s)\n", backend_name.c_str(), e.what());
    }
  }
}

int main() {
  printf("=== MLLib Advanced GPU Usage Verification ===\n");
  printf("This test verifies GPU computation effectiveness with large matrices\n\n");
  
  testMatrixMultiplication();
  testBackendSelection();
  
  printf("\n=== GPU Performance Test Complete ===\n");
  printf("\nKey Findings:\n");
  printf("- Check if GPU shows advantages for larger matrices (1024x1024+)\n");
  printf("- Small performance differences may be due to GPU overhead\n"); 
  printf("- Metal backend is optimized for Apple Silicon GPUs\n");
  
  return 0;
}
