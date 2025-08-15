/**
 * @file cuda_kernels.hpp
 * @brief Header for CUDA kernel implementations
 */

#pragma once

#include <cstddef>

namespace MLLib {
namespace backend {
namespace cuda {

/**
 * @brief Initialize CUDA context and cuBLAS
 */
void cuda_init();

/**
 * @brief Cleanup CUDA context
 */
void cuda_cleanup();

/**
 * @brief Check if CUDA is available on the system
 * @return true if CUDA devices are available, false otherwise
 */
bool cuda_is_available();

/**
 * @brief Get GPU memory information
 * @param free_bytes Pointer to store free memory in bytes
 * @param total_bytes Pointer to store total memory in bytes
 */
void cuda_get_memory_info(size_t* free_bytes, size_t* total_bytes);

/**
 * @brief GPU matrix multiplication using cuBLAS
 * @param h_a Host pointer to matrix A (m x k)
 * @param h_b Host pointer to matrix B (k x n)
 * @param h_c Host pointer to result matrix C (m x n)
 * @param m Number of rows in A and C
 * @param n Number of columns in B and C
 * @param k Number of columns in A and rows in B
 */
void cuda_matmul(const double* h_a, const double* h_b, double* h_c,
                 int m, int n, int k);

/**
 * @brief GPU element-wise addition
 * @param h_a Host pointer to first array
 * @param h_b Host pointer to second array
 * @param h_result Host pointer to result array
 * @param size Number of elements
 */
void cuda_add(const double* h_a, const double* h_b, double* h_result, size_t size);

/**
 * @brief GPU element-wise subtraction
 * @param h_a Host pointer to first array
 * @param h_b Host pointer to second array
 * @param h_result Host pointer to result array
 * @param size Number of elements
 */
void cuda_subtract(const double* h_a, const double* h_b, double* h_result, size_t size);

/**
 * @brief GPU element-wise multiplication
 * @param h_a Host pointer to first array
 * @param h_b Host pointer to second array
 * @param h_result Host pointer to result array
 * @param size Number of elements
 */
void cuda_multiply(const double* h_a, const double* h_b, double* h_result, size_t size);

/**
 * @brief GPU scalar addition
 * @param h_a Host pointer to input array
 * @param scalar Scalar value to add
 * @param h_result Host pointer to result array
 * @param size Number of elements
 */
void cuda_add_scalar(const double* h_a, double scalar, double* h_result, size_t size);

/**
 * @brief GPU scalar multiplication
 * @param h_a Host pointer to input array
 * @param scalar Scalar value to multiply
 * @param h_result Host pointer to result array
 * @param size Number of elements
 */
void cuda_multiply_scalar(const double* h_a, double scalar, double* h_result, size_t size);

/**
 * @brief GPU fill array with value
 * @param h_array Host pointer to array to fill
 * @param value Value to fill with
 * @param size Number of elements
 */
void cuda_fill(double* h_array, double value, size_t size);

/**
 * @brief GPU copy array
 * @param h_src Host pointer to source array
 * @param h_dst Host pointer to destination array
 * @param size Number of elements
 */
void cuda_copy(const double* h_src, double* h_dst, size_t size);

}  // namespace cuda
}  // namespace backend
}  // namespace MLLib
