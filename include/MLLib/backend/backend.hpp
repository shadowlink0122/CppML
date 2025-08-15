#pragma once

#include "../device/device.hpp"
#include "../ndarray.hpp"

/**
 * @file backend.hpp
 * @brief Backend interface for CPU/GPU computation
 */

namespace MLLib {
namespace backend {

/**
 * @class Backend
 * @brief Backend interface for device-specific operations
 */
class Backend {
public:
  /**
   * @brief Matrix multiplication
   * @param a Left matrix [m, k]
   * @param b Right matrix [k, n]
   * @param result Output matrix [m, n]
   */
  static void matmul(const NDArray& a, const NDArray& b, NDArray& result);

  /**
   * @brief Element-wise addition
   * @param a First array
   * @param b Second array
   * @param result Output array
   */
  static void add(const NDArray& a, const NDArray& b, NDArray& result);

  /**
   * @brief Element-wise subtraction
   * @param a First array
   * @param b Second array
   * @param result Output array (a - b)
   */
  static void subtract(const NDArray& a, const NDArray& b, NDArray& result);

  /**
   * @brief Element-wise multiplication
   * @param a First array
   * @param b Second array
   * @param result Output array
   */
  static void multiply(const NDArray& a, const NDArray& b, NDArray& result);

  /**
   * @brief Scalar addition
   * @param a Array
   * @param scalar Scalar value
   * @param result Output array
   */
  static void add_scalar(const NDArray& a, double scalar, NDArray& result);

  /**
   * @brief Scalar multiplication
   * @param a Array
   * @param scalar Scalar value
   * @param result Output array
   */
  static void multiply_scalar(const NDArray& a, double scalar, NDArray& result);

  /**
   * @brief Fill array with value
   * @param array Array to fill
   * @param value Fill value
   */
  static void fill(NDArray& array, double value);

  /**
   * @brief Copy array data
   * @param src Source array
   * @param dst Destination array
   */
  static void copy(const NDArray& src, NDArray& dst);

private:
  // CPU-specific implementations
  static void cpu_matmul(const NDArray& a, const NDArray& b, NDArray& result);
  static void cpu_add(const NDArray& a, const NDArray& b, NDArray& result);
  static void cpu_subtract(const NDArray& a, const NDArray& b, NDArray& result);
  static void cpu_multiply(const NDArray& a, const NDArray& b, NDArray& result);
  static void cpu_add_scalar(const NDArray& a, double scalar, NDArray& result);
  static void cpu_multiply_scalar(const NDArray& a, double scalar,
                                  NDArray& result);
  static void cpu_fill(NDArray& array, double value);
  static void cpu_copy(const NDArray& src, NDArray& dst);

  // GPU-specific implementations
  static void gpu_matmul(const NDArray& a, const NDArray& b, NDArray& result);
  static void gpu_add(const NDArray& a, const NDArray& b, NDArray& result);
  static void gpu_subtract(const NDArray& a, const NDArray& b, NDArray& result);
  static void gpu_multiply(const NDArray& a, const NDArray& b, NDArray& result);
  static void gpu_add_scalar(const NDArray& a, double scalar, NDArray& result);
  static void gpu_multiply_scalar(const NDArray& a, double scalar,
                                  NDArray& result);
  static void gpu_fill(NDArray& array, double value);
  static void gpu_copy(const NDArray& src, NDArray& dst);
};

}  // namespace backend
}  // namespace MLLib
