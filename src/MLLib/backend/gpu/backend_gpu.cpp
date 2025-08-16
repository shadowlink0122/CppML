#include "../../../../include/MLLib/backend/backend.hpp"
#include "../../../../include/MLLib/ndarray.hpp"
#include <iostream>
#include <stdexcept>

#ifdef WITH_CUDA
#include "cuda_kernels.hpp"
#endif

/**
 * @file backend_gpu.cpp
 * @brief GPU backend implementation for MLLib
 *
 * This implementation uses CUDA when available, with automatic fallback
 * to CPU implementation when CUDA is not available or initialization fails.
 *
 * Features:
 * - CUDA kernels for high-performance GPU computation
 * - cuBLAS integration for optimized matrix operations
 * - Automatic CUDA availability detection
 * - Graceful fallback to CPU when needed
 * - Comprehensive error handling and memory management
 */

namespace MLLib {
namespace Backend {

// Helper function to check CUDA availability and handle fallback
bool use_cuda() {
  static bool cuda_checked = false;
  static bool cuda_available = false;

  if (!cuda_checked) {
#ifdef WITH_CUDA
    try {
      cuda_available = cuda::cuda_is_available();
      if (cuda_available) {
        cuda::cuda_init();
        std::cout << "GPU backend: CUDA initialized successfully" << std::endl;
      } else {
        std::cout << "GPU backend: CUDA not available, using CPU fallback"
                  << std::endl;
      }
    } catch (const std::exception& e) {
      std::cout << "GPU backend: CUDA initialization failed (" << e.what()
                << "), using CPU fallback" << std::endl;
      cuda_available = false;
    }
#else
    std::cout
        << "GPU backend: Compiled without CUDA support, using CPU fallback"
        << std::endl;
    cuda_available = false;
#endif
    cuda_checked = true;
  }

  return cuda_available;
}

// GPU matrix multiplication implementation
void Backend::gpu_matmul(const NDArray& a, const NDArray& b, NDArray& result) {
  if (a.shape().size() != 2 || b.shape().size() != 2) {
    throw std::invalid_argument("Matrix multiplication requires 2D arrays");
  }

  size_t m = a.shape()[0];
  size_t k = a.shape()[1];
  size_t n = b.shape()[1];

  if (k != b.shape()[0]) {
    throw std::invalid_argument("Inner dimensions must match");
  }

  // Ensure result has correct shape
  if (result.shape().size() != 2 || result.shape()[0] != m ||
      result.shape()[1] != n) {
    result = NDArray({m, n});
  }

  const double* a_data = a.data();
  const double* b_data = b.data();
  double* result_data = result.data();

  if (use_cuda()) {
#ifdef WITH_CUDA
    try {
      cuda::cuda_matmul(a_data, b_data, result_data, static_cast<int>(m),
                        static_cast<int>(n), static_cast<int>(k));
      return;
    } catch (const std::exception& e) {
      std::cout << "GPU matmul failed, falling back to CPU: " << e.what()
                << std::endl;
    }
#endif
  }

  // CPU fallback
  cpu_matmul(a, b, result);
}

// GPU element-wise addition
void Backend::gpu_add(const NDArray& a, const NDArray& b, NDArray& result) {
  if (a.shape() != b.shape()) {
    throw std::invalid_argument("Shapes must match for addition");
  }

  if (result.shape() != a.shape()) {
    result = NDArray(a.shape());
  }

  const double* a_data = a.data();
  const double* b_data = b.data();
  double* result_data = result.data();
  size_t size = a.size();

  if (use_cuda()) {
#ifdef WITH_CUDA
    try {
      cuda::cuda_add(a_data, b_data, result_data, size);
      return;
    } catch (const std::exception& e) {
      std::cout << "GPU add failed, falling back to CPU: " << e.what()
                << std::endl;
    }
#endif
  }

  // CPU fallback
  cpu_add(a, b, result);
}

// GPU element-wise subtraction
void Backend::gpu_subtract(const NDArray& a, const NDArray& b,
                           NDArray& result) {
  if (a.shape() != b.shape()) {
    throw std::invalid_argument("Shapes must match for subtraction");
  }

  if (result.shape() != a.shape()) {
    result = NDArray(a.shape());
  }

  const double* a_data = a.data();
  const double* b_data = b.data();
  double* result_data = result.data();
  size_t size = a.size();

  if (use_cuda()) {
#ifdef WITH_CUDA
    try {
      cuda::cuda_subtract(a_data, b_data, result_data, size);
      return;
    } catch (const std::exception& e) {
      std::cout << "GPU subtract failed, falling back to CPU: " << e.what()
                << std::endl;
    }
#endif
  }

  // CPU fallback
  cpu_subtract(a, b, result);
}

// GPU element-wise multiplication
void Backend::gpu_multiply(const NDArray& a, const NDArray& b,
                           NDArray& result) {
  if (a.shape() != b.shape()) {
    throw std::invalid_argument("Shapes must match for multiplication");
  }

  if (result.shape() != a.shape()) {
    result = NDArray(a.shape());
  }

  const double* a_data = a.data();
  const double* b_data = b.data();
  double* result_data = result.data();
  size_t size = a.size();

  if (use_cuda()) {
#ifdef WITH_CUDA
    try {
      cuda::cuda_multiply(a_data, b_data, result_data, size);
      return;
    } catch (const std::exception& e) {
      std::cout << "GPU multiply failed, falling back to CPU: " << e.what()
                << std::endl;
    }
#endif
  }

  // CPU fallback
  cpu_multiply(a, b, result);
}

// GPU scalar addition
void Backend::gpu_add_scalar(const NDArray& a, double scalar, NDArray& result) {
  if (result.shape() != a.shape()) {
    result = NDArray(a.shape());
  }

  const double* a_data = a.data();
  double* result_data = result.data();
  size_t size = a.size();

  if (use_cuda()) {
#ifdef WITH_CUDA
    try {
      cuda::cuda_add_scalar(a_data, scalar, result_data, size);
      return;
    } catch (const std::exception& e) {
      std::cout << "GPU add_scalar failed, falling back to CPU: " << e.what()
                << std::endl;
    }
#endif
  }

  // CPU fallback
  cpu_add_scalar(a, scalar, result);
}

// GPU scalar multiplication
void Backend::gpu_multiply_scalar(const NDArray& a, double scalar,
                                  NDArray& result) {
  if (result.shape() != a.shape()) {
    result = NDArray(a.shape());
  }

  const double* a_data = a.data();
  double* result_data = result.data();
  size_t size = a.size();

  if (use_cuda()) {
#ifdef WITH_CUDA
    try {
      cuda::cuda_multiply_scalar(a_data, scalar, result_data, size);
      return;
    } catch (const std::exception& e) {
      std::cout << "GPU multiply_scalar failed, falling back to CPU: "
                << e.what() << std::endl;
    }
#endif
  }

  // CPU fallback
  cpu_multiply_scalar(a, scalar, result);
}

// GPU fill array
void Backend::gpu_fill(NDArray& array, double value) {
  double* data = array.data();
  size_t size = array.size();

  if (use_cuda()) {
#ifdef WITH_CUDA
    try {
      cuda::cuda_fill(data, value, size);
      return;
    } catch (const std::exception& e) {
      std::cout << "GPU fill failed, falling back to CPU: " << e.what()
                << std::endl;
    }
#endif
  }

  // CPU fallback
  cpu_fill(array, value);
}

// GPU copy array
void Backend::gpu_copy(const NDArray& src, NDArray& dst) {
  if (dst.shape() != src.shape()) {
    dst = NDArray(src.shape());
  }

  const double* src_data = src.data();
  double* dst_data = dst.data();
  size_t size = src.size();

  if (use_cuda()) {
#ifdef WITH_CUDA
    try {
      cuda::cuda_copy(src_data, dst_data, size);
      return;
    } catch (const std::exception& e) {
      std::cout << "GPU copy failed, falling back to CPU: " << e.what()
                << std::endl;
    }
#endif
  }

  // CPU fallback
  cpu_copy(src, dst);
}

}  // namespace Backend
}  // namespace MLLib
