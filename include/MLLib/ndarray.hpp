#pragma once

#include <initializer_list>
#include <memory>
#include <vector>

/**
 * @file ndarray.hpp
 * @brief Multi-dimensional array implementation
 */

namespace MLLib {

/**
 * @class NDArray
 * @brief Multi-dimensional array class for tensor operations
 */
class NDArray {
public:
  /**
   * @brief Default constructor
   */
  NDArray() = default;

  /**
   * @brief Constructor with shape
   * @param shape Shape of the array
   */
  explicit NDArray(const std::vector<size_t>& shape);

  /**
   * @brief Constructor with initializer list (for convenience)
   * @param shape Shape of the array
   */
  NDArray(std::initializer_list<size_t> shape);

  /**
   * @brief Constructor from 1D vector
   * @param data 1D vector data
   */
  explicit NDArray(const std::vector<double>& data);

  /**
   * @brief Constructor from 2D vector
   * @param data 2D vector data
   */
  explicit NDArray(const std::vector<std::vector<double>>& data);

  /**
   * @brief Copy constructor
   */
  NDArray(const NDArray& other);

  /**
   * @brief Assignment operator
   */
  NDArray& operator=(const NDArray& other);

  /**
   * @brief Get element at index (1D)
   * @param index Index
   * @return Reference to element
   */
  double& operator[](size_t index);

  /**
   * @brief Get element at index (1D) - const version
   * @param index Index
   * @return Const reference to element
   */
  const double& operator[](size_t index) const;

  /**
   * @brief Get element at multi-dimensional index
   * @param indices Multi-dimensional indices
   * @return Reference to element
   */
  double& at(const std::vector<size_t>& indices);

  /**
   * @brief Get element at multi-dimensional index - const version
   * @param indices Multi-dimensional indices
   * @return Const reference to element
   */
  const double& at(const std::vector<size_t>& indices) const;

  /**
   * @brief Get shape of the array
   * @return Shape vector
   */
  const std::vector<size_t>& shape() const { return shape_; }

  /**
   * @brief Get total number of elements
   * @return Total size
   */
  size_t size() const { return size_; }

  /**
   * @brief Get raw data pointer
   * @return Raw data pointer
   */
  double* data() { return data_.get(); }

  /**
   * @brief Get raw data pointer - const version
   * @return Const raw data pointer
   */
  const double* data() const { return data_.get(); }

  /**
   * @brief Reshape the array
   * @param new_shape New shape
   */
  void reshape(const std::vector<size_t>& new_shape);

  /**
   * @brief Fill array with value
   * @param value Fill value
   */
  void fill(double value);

  /**
   * @brief Convert to 1D vector
   * @return 1D vector representation
   */
  std::vector<double> to_vector() const;

  /**
   * @brief Matrix multiplication (for 2D arrays)
   * @param other Other NDArray
   * @return Result of matrix multiplication
   */
  NDArray matmul(const NDArray& other) const;

  /**
   * @brief Element-wise addition
   * @param other Other NDArray
   * @return Result of addition
   */
  NDArray operator+(const NDArray& other) const;

  /**
   * @brief Element-wise subtraction
   * @param other Other NDArray
   * @return Result of subtraction
   */
  NDArray operator-(const NDArray& other) const;

  /**
   * @brief Element-wise multiplication
   * @param other Other NDArray
   * @return Result of multiplication
   */
  NDArray operator*(const NDArray& other) const;

  /**
   * @brief Scalar addition
   * @param scalar Scalar value
   * @return Result of scalar addition
   */
  NDArray operator+(double scalar) const;

  /**
   * @brief Scalar multiplication
   * @param scalar Scalar value
   * @return Result of scalar multiplication
   */
  NDArray operator*(double scalar) const;

private:
  std::vector<size_t> shape_;
  size_t size_ = 0;
  std::unique_ptr<double[]> data_;

  /**
   * @brief Calculate total size from shape
   */
  void calculate_size();

  /**
   * @brief Convert multi-dimensional index to linear index
   * @param indices Multi-dimensional indices
   * @return Linear index
   */
  size_t to_linear_index(const std::vector<size_t>& indices) const;
};

}  // namespace MLLib
