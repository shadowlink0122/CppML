#include "../../../../include/MLLib/backend/backend.hpp"
#include "../../../../include/MLLib/ndarray.hpp"
#include <stdexcept>

namespace MLLib {
namespace backend {

// CPU matrix multiplication implementation
void Backend::cpu_matmul(const NDArray& a, const NDArray& b, NDArray& result) {
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

  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < n; ++j) {
      double sum = 0.0;
      for (size_t l = 0; l < k; ++l) {
        sum += a_data[i * k + l] * b_data[l * n + j];
      }
      result_data[i * n + j] = sum;
    }
  }
}

// CPU element-wise addition
void Backend::cpu_add(const NDArray& a, const NDArray& b, NDArray& result) {
  if (a.shape() != b.shape()) {
    throw std::invalid_argument("Shapes must match for addition");
  }

  if (result.shape() != a.shape()) {
    result = NDArray(a.shape());
  }

  const double* a_data = a.data();
  const double* b_data = b.data();
  double* result_data = result.data();

  for (size_t i = 0; i < a.size(); ++i) {
    result_data[i] = a_data[i] + b_data[i];
  }
}

// CPU element-wise subtraction
void Backend::cpu_subtract(const NDArray& a, const NDArray& b,
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

  for (size_t i = 0; i < a.size(); ++i) {
    result_data[i] = a_data[i] - b_data[i];
  }
}

// CPU element-wise multiplication
void Backend::cpu_multiply(const NDArray& a, const NDArray& b,
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

  for (size_t i = 0; i < a.size(); ++i) {
    result_data[i] = a_data[i] * b_data[i];
  }
}

// CPU scalar addition
void Backend::cpu_add_scalar(const NDArray& a, double scalar, NDArray& result) {
  if (result.shape() != a.shape()) {
    result = NDArray(a.shape());
  }

  const double* a_data = a.data();
  double* result_data = result.data();

  for (size_t i = 0; i < a.size(); ++i) {
    result_data[i] = a_data[i] + scalar;
  }
}

// CPU scalar multiplication
void Backend::cpu_multiply_scalar(const NDArray& a, double scalar,
                                  NDArray& result) {
  if (result.shape() != a.shape()) {
    result = NDArray(a.shape());
  }

  const double* a_data = a.data();
  double* result_data = result.data();

  for (size_t i = 0; i < a.size(); ++i) {
    result_data[i] = a_data[i] * scalar;
  }
}

// CPU fill array
void Backend::cpu_fill(NDArray& array, double value) {
  double* data = array.data();
  for (size_t i = 0; i < array.size(); ++i) {
    data[i] = value;
  }
}

// CPU copy array
void Backend::cpu_copy(const NDArray& src, NDArray& dst) {
  if (dst.shape() != src.shape()) {
    dst = NDArray(src.shape());
  }

  const double* src_data = src.data();
  double* dst_data = dst.data();

  for (size_t i = 0; i < src.size(); ++i) {
    dst_data[i] = src_data[i];
  }
}

}  // namespace backend
}  // namespace MLLib
