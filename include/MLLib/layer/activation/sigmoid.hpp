#pragma once

#include "activation.hpp"

/**
 * @file sigmoid.hpp
 * @brief Sigmoid activation function implementation
 */

namespace MLLib {
namespace layer {
namespace activation {

/**
 * @class Sigmoid
 * @brief Sigmoid activation function
 */
class Sigmoid : public Activation {
public:
  /**
   * @brief Constructor
   */
  Sigmoid() = default;

  /**
   * @brief Destructor
   */
  virtual ~Sigmoid() = default;

  /**
   * @brief Forward propagation
   * @param input Input data
   * @return Output data (1 / (1 + exp(-input)))
   */
  NDArray forward(const NDArray& input) override;

  /**
   * @brief Backward propagation
   * @param grad_output Gradient from the next layer
   * @return Gradient with respect to input
   */
  NDArray backward(const NDArray& grad_output) override;

private:
  NDArray last_output_;  ///< Cache output for backward pass
};

}  // namespace activation
}  // namespace layer
}  // namespace MLLib
