#pragma once

#include "activation.hpp"

/**
 * @file elu.hpp
 * @brief ELU activation function implementation
 */

namespace MLLib {
namespace layer {
namespace activation {

/**
 * @class ELU
 * @brief Exponential Linear Unit activation function
 *
 * ELU has negative values which allows them to push mean unit activations
 * closer to zero: f(x) = x if x > 0, alpha * (exp(x) - 1) if x <= 0
 */
class ELU : public Activation {
public:
  /**
   * @brief Constructor
   * @param alpha Controls the value to which an ELU saturates for negative
   * inputs (default: 1.0)
   */
  explicit ELU(double alpha = 1.0);

  /**
   * @brief Destructor
   */
  virtual ~ELU() = default;

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
  double alpha_;  ///< ELU parameter
};

}  // namespace activation
}  // namespace layer
}  // namespace MLLib
