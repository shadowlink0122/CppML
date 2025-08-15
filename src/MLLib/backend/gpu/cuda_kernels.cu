/**
 * @file cuda_kernels.cu
 * @brief CUDA kernel implementations for GPU backend
 */

#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <stdexcept>
#include <string>

namespace MLLib {
namespace Backend {
namespace cuda {

// Error checking macro
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t error = call;                                                  \
    if (error != cudaSuccess) {                                                \
      throw std::runtime_error("CUDA error: " +                                \
                               std::string(cudaGetErrorString(error)));        \
    }                                                                          \
  } while (0)

#define CUBLAS_CHECK(call)                                                     \
  do {                                                                         \
    cublasStatus_t status = call;                                              \
    if (status != CUBLAS_STATUS_SUCCESS) {                                     \
      throw std::runtime_error("cuBLAS error: " + std::to_string(status));     \
    }                                                                          \
  } while (0)

// Thread block size for kernels
const int BLOCK_SIZE = 256;
const int TILE_SIZE = 16;

// CUDA kernel for element-wise addition
__global__ void add_kernel(const double* a, const double* b, double* result,
                           size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    result[idx] = a[idx] + b[idx];
  }
}

// CUDA kernel for element-wise subtraction
__global__ void subtract_kernel(const double* a, const double* b,
                                double* result, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    result[idx] = a[idx] - b[idx];
  }
}

// CUDA kernel for element-wise multiplication
__global__ void multiply_kernel(const double* a, const double* b,
                                double* result, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    result[idx] = a[idx] * b[idx];
  }
}

// CUDA kernel for scalar addition
__global__ void add_scalar_kernel(const double* a, double scalar,
                                  double* result, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    result[idx] = a[idx] + scalar;
  }
}

// CUDA kernel for scalar multiplication
__global__ void multiply_scalar_kernel(const double* a, double scalar,
                                       double* result, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    result[idx] = a[idx] * scalar;
  }
}

// CUDA kernel for filling array with value
__global__ void fill_kernel(double* array, double value, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    array[idx] = value;
  }
}

// Optimized matrix multiplication kernel using shared memory
__global__ void matmul_kernel(const double* a, const double* b, double* c,
                              int m, int n, int k) {
  __shared__ double tile_a[TILE_SIZE][TILE_SIZE];
  __shared__ double tile_b[TILE_SIZE][TILE_SIZE];

  int row = blockIdx.y * TILE_SIZE + threadIdx.y;
  int col = blockIdx.x * TILE_SIZE + threadIdx.x;

  double sum = 0.0;

  for (int tile = 0; tile < (k + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
    // Load tiles into shared memory
    if (row < m && tile * TILE_SIZE + threadIdx.x < k) {
      tile_a[threadIdx.y][threadIdx.x] =
          a[row * k + tile * TILE_SIZE + threadIdx.x];
    } else {
      tile_a[threadIdx.y][threadIdx.x] = 0.0;
    }

    if (col < n && tile * TILE_SIZE + threadIdx.y < k) {
      tile_b[threadIdx.y][threadIdx.x] =
          b[(tile * TILE_SIZE + threadIdx.y) * n + col];
    } else {
      tile_b[threadIdx.y][threadIdx.x] = 0.0;
    }

    __syncthreads();

    // Compute partial dot product
    for (int i = 0; i < TILE_SIZE; ++i) {
      sum += tile_a[threadIdx.y][i] * tile_b[i][threadIdx.x];
    }

    __syncthreads();
  }

  if (row < m && col < n) {
    c[row * n + col] = sum;
  }
}

// Global cuBLAS handle
static cublasHandle_t cublas_handle = nullptr;

// Initialize CUDA context and cuBLAS
void cuda_init() {
  if (cublas_handle == nullptr) {
    CUBLAS_CHECK(cublasCreate(&cublas_handle));

    // Check if we're in simulation mode
    const char* sim_mode = std::getenv("GPU_SIMULATION_MODE");
    if (sim_mode && std::string(sim_mode) == "1") {
      printf("CUDA simulation mode initialized successfully\n");
    } else {
      printf("CUDA context initialized successfully\n");
    }
  }
}

// Cleanup CUDA context
void cuda_cleanup() {
  if (cublas_handle != nullptr) {
    cublasDestroy(cublas_handle);
    cublas_handle = nullptr;
  }
}

// Check if CUDA is available
bool cuda_is_available() {
  // Check for GPU simulation mode
  const char* sim_mode = std::getenv("GPU_SIMULATION_MODE");
  if (sim_mode && std::string(sim_mode) == "1") {
    return true;  // Force GPU availability in simulation mode
  }

  int device_count;
  cudaError_t error = cudaGetDeviceCount(&device_count);
  return (error == cudaSuccess && device_count > 0);
}

// Get GPU memory info
void cuda_get_memory_info(size_t* free_bytes, size_t* total_bytes) {
  CUDA_CHECK(cudaMemGetInfo(free_bytes, total_bytes));
}

// GPU matrix multiplication using cuBLAS
void cuda_matmul(const double* h_a, const double* h_b, double* h_c, int m,
                 int n, int k) {
  cuda_init();

  size_t size_a = m * k * sizeof(double);
  size_t size_b = k * n * sizeof(double);
  size_t size_c = m * n * sizeof(double);

  double* d_a = nullptr;
  double* d_b = nullptr;
  double* d_c = nullptr;

  try {
    // Allocate GPU memory
    CUDA_CHECK(cudaMalloc(&d_a, size_a));
    CUDA_CHECK(cudaMalloc(&d_b, size_b));
    CUDA_CHECK(cudaMalloc(&d_c, size_c));

    // Copy data to GPU
    CUDA_CHECK(cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice));

    // Perform matrix multiplication using cuBLAS
    const double alpha = 1.0, beta = 0.0;
    CUBLAS_CHECK(cublasDgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
                             &alpha, d_b, n, d_a, k, &beta, d_c, n));

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_c, d_c, size_c, cudaMemcpyDeviceToHost));

  } catch (...) {
    // Cleanup on error
    if (d_a) cudaFree(d_a);
    if (d_b) cudaFree(d_b);
    if (d_c) cudaFree(d_c);
    throw;
  }

  // Cleanup
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}

// GPU element-wise addition
void cuda_add(const double* h_a, const double* h_b, double* h_result,
              size_t size) {
  double* d_a = nullptr;
  double* d_b = nullptr;
  double* d_result = nullptr;

  size_t byte_size = size * sizeof(double);

  try {
    // Allocate GPU memory
    CUDA_CHECK(cudaMalloc(&d_a, byte_size));
    CUDA_CHECK(cudaMalloc(&d_b, byte_size));
    CUDA_CHECK(cudaMalloc(&d_result, byte_size));

    // Copy data to GPU
    CUDA_CHECK(cudaMemcpy(d_a, h_a, byte_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, byte_size, cudaMemcpyHostToDevice));

    // Launch kernel
    int grid_size = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    add_kernel<<<grid_size, BLOCK_SIZE>>>(d_a, d_b, d_result, size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back
    CUDA_CHECK(
        cudaMemcpy(h_result, d_result, byte_size, cudaMemcpyDeviceToHost));

  } catch (...) {
    if (d_a) cudaFree(d_a);
    if (d_b) cudaFree(d_b);
    if (d_result) cudaFree(d_result);
    throw;
  }

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_result);
}

// GPU element-wise subtraction
void cuda_subtract(const double* h_a, const double* h_b, double* h_result,
                   size_t size) {
  double* d_a = nullptr;
  double* d_b = nullptr;
  double* d_result = nullptr;

  size_t byte_size = size * sizeof(double);

  try {
    CUDA_CHECK(cudaMalloc(&d_a, byte_size));
    CUDA_CHECK(cudaMalloc(&d_b, byte_size));
    CUDA_CHECK(cudaMalloc(&d_result, byte_size));

    CUDA_CHECK(cudaMemcpy(d_a, h_a, byte_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, byte_size, cudaMemcpyHostToDevice));

    int grid_size = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    subtract_kernel<<<grid_size, BLOCK_SIZE>>>(d_a, d_b, d_result, size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(
        cudaMemcpy(h_result, d_result, byte_size, cudaMemcpyDeviceToHost));

  } catch (...) {
    if (d_a) cudaFree(d_a);
    if (d_b) cudaFree(d_b);
    if (d_result) cudaFree(d_result);
    throw;
  }

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_result);
}

// GPU element-wise multiplication
void cuda_multiply(const double* h_a, const double* h_b, double* h_result,
                   size_t size) {
  double* d_a = nullptr;
  double* d_b = nullptr;
  double* d_result = nullptr;

  size_t byte_size = size * sizeof(double);

  try {
    CUDA_CHECK(cudaMalloc(&d_a, byte_size));
    CUDA_CHECK(cudaMalloc(&d_b, byte_size));
    CUDA_CHECK(cudaMalloc(&d_result, byte_size));

    CUDA_CHECK(cudaMemcpy(d_a, h_a, byte_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, byte_size, cudaMemcpyHostToDevice));

    int grid_size = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    multiply_kernel<<<grid_size, BLOCK_SIZE>>>(d_a, d_b, d_result, size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(
        cudaMemcpy(h_result, d_result, byte_size, cudaMemcpyDeviceToHost));

  } catch (...) {
    if (d_a) cudaFree(d_a);
    if (d_b) cudaFree(d_b);
    if (d_result) cudaFree(d_result);
    throw;
  }

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_result);
}

// GPU scalar addition
void cuda_add_scalar(const double* h_a, double scalar, double* h_result,
                     size_t size) {
  double* d_a = nullptr;
  double* d_result = nullptr;

  size_t byte_size = size * sizeof(double);

  try {
    CUDA_CHECK(cudaMalloc(&d_a, byte_size));
    CUDA_CHECK(cudaMalloc(&d_result, byte_size));

    CUDA_CHECK(cudaMemcpy(d_a, h_a, byte_size, cudaMemcpyHostToDevice));

    int grid_size = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    add_scalar_kernel<<<grid_size, BLOCK_SIZE>>>(d_a, scalar, d_result, size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(
        cudaMemcpy(h_result, d_result, byte_size, cudaMemcpyDeviceToHost));

  } catch (...) {
    if (d_a) cudaFree(d_a);
    if (d_result) cudaFree(d_result);
    throw;
  }

  cudaFree(d_a);
  cudaFree(d_result);
}

// GPU scalar multiplication
void cuda_multiply_scalar(const double* h_a, double scalar, double* h_result,
                          size_t size) {
  double* d_a = nullptr;
  double* d_result = nullptr;

  size_t byte_size = size * sizeof(double);

  try {
    CUDA_CHECK(cudaMalloc(&d_a, byte_size));
    CUDA_CHECK(cudaMalloc(&d_result, byte_size));

    CUDA_CHECK(cudaMemcpy(d_a, h_a, byte_size, cudaMemcpyHostToDevice));

    int grid_size = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    multiply_scalar_kernel<<<grid_size, BLOCK_SIZE>>>(d_a, scalar, d_result,
                                                      size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(
        cudaMemcpy(h_result, d_result, byte_size, cudaMemcpyDeviceToHost));

  } catch (...) {
    if (d_a) cudaFree(d_a);
    if (d_result) cudaFree(d_result);
    throw;
  }

  cudaFree(d_a);
  cudaFree(d_result);
}

// GPU fill array
void cuda_fill(double* h_array, double value, size_t size) {
  double* d_array = nullptr;
  size_t byte_size = size * sizeof(double);

  try {
    CUDA_CHECK(cudaMalloc(&d_array, byte_size));

    int grid_size = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    fill_kernel<<<grid_size, BLOCK_SIZE>>>(d_array, value, size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_array, d_array, byte_size, cudaMemcpyDeviceToHost));

  } catch (...) {
    if (d_array) cudaFree(d_array);
    throw;
  }

  cudaFree(d_array);
}

// GPU copy array
void cuda_copy(const double* h_src, double* h_dst, size_t size) {
  size_t byte_size = size * sizeof(double);

  // For simple copy, we can use cudaMemcpy directly
  // or use GPU kernel for consistency
  double* d_src = nullptr;
  double* d_dst = nullptr;

  try {
    CUDA_CHECK(cudaMalloc(&d_src, byte_size));
    CUDA_CHECK(cudaMalloc(&d_dst, byte_size));

    CUDA_CHECK(cudaMemcpy(d_src, h_src, byte_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dst, d_src, byte_size, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(h_dst, d_dst, byte_size, cudaMemcpyDeviceToHost));

  } catch (...) {
    if (d_src) cudaFree(d_src);
    if (d_dst) cudaFree(d_dst);
    throw;
  }

  cudaFree(d_src);
  cudaFree(d_dst);
}

}  // namespace cuda
}  // namespace Backend
}  // namespace MLLib
