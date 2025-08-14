#pragma once

#include "activation.hpp"

/**
 * @file tanh.hpp
 * @brief Hyperbolic tangent activation function implementation
 */

namespace MLLib {
namespace layer {
namespace activation {

/**
 * @class Tanh
 * @brief Hyperbolic tangent activation function
 *
 * Output range: (-1, 1)
 * Derivative: 1 - tanh²(x)
 */
class Tanh : public Activation {
public:
  /**
   * @brief Constructor
   */
  Tanh() = default;

  /**
   * @brief Destructor
   */
  virtual ~Tanh() = default;

  /**
   * @brief Forward pass
   * @param input Input tensor
   * @return Output tensor
   */
  NDArray forward(const NDArray& input) override;

  /**
   * @brief Backward pass
   * @param grad_output Gradient from next layer
   * @return Gradient with respect to input
   */
  NDArray backward(const NDArray& grad_output) override;
};

}  // namespace activation
}  // namespace layer
}  // namespace MLLib
