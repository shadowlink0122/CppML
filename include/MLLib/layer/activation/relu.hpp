#pragma once

#include "activation.hpp"

/**
 * @file relu.hpp
 * @brief ReLU activation function implementation
 */

namespace MLLib {
namespace layer {
namespace activation {

/**
 * @class ReLU
 * @brief Rectified Linear Unit activation function
 */
class ReLU : public Activation {
public:
  /**
   * @brief Constructor
   */
  ReLU() = default;

  /**
   * @brief Destructor
   */
  virtual ~ReLU() = default;

  /**
   * @brief Forward propagation
   * @param input Input data
   * @return Output data (max(0, input))
   */
  NDArray forward(const NDArray& input) override;

  /**
   * @brief Backward propagation
   * @param grad_output Gradient from the next layer
   * @return Gradient with respect to input
   */
  NDArray backward(const NDArray& grad_output) override;
};

}  // namespace activation
}  // namespace layer
}  // namespace MLLib
