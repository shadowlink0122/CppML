#pragma once

#include "activation.hpp"

/**
 * @file softmax.hpp
 * @brief Softmax activation function implementation
 */

namespace MLLib {
namespace layer {
namespace activation {

/**
 * @class Softmax
 * @brief Softmax activation function
 * 
 * Softmax is commonly used in the output layer for multi-class classification:
 * f(x_i) = exp(x_i) / sum(exp(x_j)) for all j
 * Numerically stable implementation using the log-sum-exp trick
 */
class Softmax : public Activation {
public:
  /**
   * @brief Constructor
   * @param axis Axis along which to apply softmax (default: -1, last axis)
   */
  explicit Softmax(int axis = -1);

  /**
   * @brief Destructor
   */
  virtual ~Softmax() = default;

  /**
   * @brief Forward propagation
   * @param input Input data
   * @return Output data (probabilities)
   */
  NDArray forward(const NDArray& input) override;

  /**
   * @brief Backward propagation
   * @param grad_output Gradient from the next layer
   * @return Gradient with respect to input
   */
  NDArray backward(const NDArray& grad_output) override;

  /**
   * @brief Get axis parameter
   * @return Axis value
   */
  int get_axis() const { return axis_; }

private:
  int axis_;           ///< Axis along which to apply softmax
  NDArray last_output_; ///< Cache output for backward pass
};

}  // namespace activation
}  // namespace layer
}  // namespace MLLib
