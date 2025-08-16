#pragma once

#include "activation.hpp"

/**
 * @file leaky_relu.hpp
 * @brief Leaky ReLU activation function implementation
 */

namespace MLLib {
namespace layer {
namespace activation {

/**
 * @class LeakyReLU
 * @brief Leaky Rectified Linear Unit activation function
 * 
 * LeakyReLU allows a small positive gradient when the unit is not active:
 * f(x) = max(alpha * x, x) where alpha is a small positive constant
 */
class LeakyReLU : public Activation {
public:
  /**
   * @brief Constructor
   * @param alpha Negative slope coefficient (default: 0.01)
   */
  explicit LeakyReLU(double alpha = 0.01);

  /**
   * @brief Destructor
   */
  virtual ~LeakyReLU() = default;

  /**
   * @brief Forward propagation
   * @param input Input data
   * @return Output data
   */
  NDArray forward(const NDArray& input) override;

  /**
   * @brief Backward propagation
   * @param grad_output Gradient from the next layer
   * @return Gradient with respect to input
   */
  NDArray backward(const NDArray& grad_output) override;

  /**
   * @brief Get alpha parameter
   * @return Alpha value
   */
  double get_alpha() const { return alpha_; }

private:
  double alpha_;  ///< Negative slope coefficient
};

}  // namespace activation
}  // namespace layer
}  // namespace MLLib
