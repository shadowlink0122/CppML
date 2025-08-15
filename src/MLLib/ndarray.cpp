#include "../../include/MLLib/ndarray.hpp"
#include <algorithm>
#include <cstring>
#include <numeric>
#include <stdexcept>

namespace MLLib {

NDArray::NDArray(const std::vector<size_t>& shape) : shape_(shape) {
  calculate_size();
  data_ = std::make_unique<double[]>(size_);
  std::fill(data_.get(), data_.get() + size_, 0.0);
}

NDArray::NDArray(std::initializer_list<size_t> shape) : shape_(shape) {
  calculate_size();
  data_ = std::make_unique<double[]>(size_);
  std::fill(data_.get(), data_.get() + size_, 0.0);
}

NDArray::NDArray(const std::vector<double>& data) {
  shape_ = {data.size()};
  calculate_size();
  data_ = std::make_unique<double[]>(size_);
  std::copy(data.begin(), data.end(), data_.get());
}

NDArray::NDArray(const std::vector<std::vector<double>>& data) {
  if (data.empty()) {
    shape_ = {0, 0};
    size_ = 0;
    data_ = nullptr;
    return;
  }

  size_t rows = data.size();
  size_t cols = data[0].size();
  shape_ = {rows, cols};
  calculate_size();

  data_ = std::make_unique<double[]>(size_);
  for (size_t i = 0; i < rows; ++i) {
    if (data[i].size() != cols) {
      throw std::invalid_argument(
          "All rows must have the same number of columns");
    }
    std::copy(data[i].begin(), data[i].end(), data_.get() + i * cols);
  }
}

NDArray::NDArray(const NDArray& other)
    : shape_(other.shape_), size_(other.size_) {
  if (size_ > 0) {
    data_ = std::make_unique<double[]>(size_);
    std::copy(other.data_.get(), other.data_.get() + size_, data_.get());
  }
}

NDArray& NDArray::operator=(const NDArray& other) {
  if (this != &other) {
    shape_ = other.shape_;
    size_ = other.size_;
    if (size_ > 0) {
      data_ = std::make_unique<double[]>(size_);
      std::copy(other.data_.get(), other.data_.get() + size_, data_.get());
    } else {
      data_.reset();
    }
  }
  return *this;
}

double& NDArray::operator[](size_t index) {
  if (index >= size_) {
    throw std::out_of_range("Index out of range");
  }
  return data_[index];
}

const double& NDArray::operator[](size_t index) const {
  if (index >= size_) {
    throw std::out_of_range("Index out of range");
  }
  return data_[index];
}

double& NDArray::at(const std::vector<size_t>& indices) {
  size_t linear_index = to_linear_index(indices);
  return data_[linear_index];
}

const double& NDArray::at(const std::vector<size_t>& indices) const {
  size_t linear_index = to_linear_index(indices);
  return data_[linear_index];
}

void NDArray::reshape(const std::vector<size_t>& new_shape) {
  size_t new_size = 1;
  for (size_t dim : new_shape) {
    new_size *= dim;
  }

  if (new_size != size_) {
    throw std::invalid_argument("New shape must have the same total size");
  }

  shape_ = new_shape;
}

void NDArray::fill(double value) {
  std::fill(data_.get(), data_.get() + size_, value);
}

std::vector<double> NDArray::to_vector() const {
  std::vector<double> result(size_);
  std::copy(data_.get(), data_.get() + size_, result.begin());
  return result;
}

NDArray NDArray::matmul(const NDArray& other) const {
  if (shape_.size() != 2 || other.shape_.size() != 2) {
    throw std::invalid_argument("Matrix multiplication requires 2D arrays");
  }

  size_t m = shape_[0];        // rows of this
  size_t k = shape_[1];        // cols of this / rows of other
  size_t n = other.shape_[1];  // cols of other

  if (k != other.shape_[0]) {
    throw std::invalid_argument(
        "Inner dimensions must match for matrix multiplication");
  }

  NDArray result({m, n});

  // Simple CPU matrix multiplication
  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < n; ++j) {
      double sum = 0.0;
      for (size_t l = 0; l < k; ++l) {
        sum += data_[i * k + l] * other.data_[l * n + j];
      }
      result.data_[i * n + j] = sum;
    }
  }

  return result;
}

NDArray NDArray::operator+(const NDArray& other) const {
  if (shape_ != other.shape_) {
    throw std::invalid_argument("Shapes must match for element-wise addition");
  }

  NDArray result(shape_);
  for (size_t i = 0; i < size_; ++i) {
    result.data_[i] = data_[i] + other.data_[i];
  }
  return result;
}

NDArray NDArray::operator-(const NDArray& other) const {
  if (shape_ != other.shape_) {
    throw std::invalid_argument(
        "Shapes must match for element-wise subtraction");
  }

  NDArray result(shape_);
  for (size_t i = 0; i < size_; ++i) {
    result.data_[i] = data_[i] - other.data_[i];
  }
  return result;
}

NDArray NDArray::operator*(const NDArray& other) const {
  if (shape_ != other.shape_) {
    throw std::invalid_argument(
        "Shapes must match for element-wise multiplication");
  }

  NDArray result(shape_);
  for (size_t i = 0; i < size_; ++i) {
    result.data_[i] = data_[i] * other.data_[i];
  }
  return result;
}

NDArray NDArray::operator+(double scalar) const {
  NDArray result(shape_);
  for (size_t i = 0; i < size_; ++i) {
    result.data_[i] = data_[i] + scalar;
  }
  return result;
}

NDArray NDArray::operator*(double scalar) const {
  NDArray result(shape_);
  for (size_t i = 0; i < size_; ++i) {
    result.data_[i] = data_[i] * scalar;
  }
  return result;
}

void NDArray::calculate_size() {
  size_ = 1;
  for (size_t dim : shape_) {
    size_ *= dim;
  }
}

size_t NDArray::to_linear_index(const std::vector<size_t>& indices) const {
  if (indices.size() != shape_.size()) {
    throw std::invalid_argument(
        "Number of indices must match number of dimensions");
  }

  size_t linear_index = 0;
  size_t stride = 1;

  for (int i = shape_.size() - 1; i >= 0; --i) {
    if (indices[i] >= shape_[i]) {
      throw std::out_of_range("Index out of range");
    }
    linear_index += indices[i] * stride;
    stride *= shape_[i];
  }

  return linear_index;
}

}  // namespace MLLib
