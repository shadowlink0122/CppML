#pragma once

#include "activation.hpp"

/**
 * @file gelu.hpp
 * @brief GELU activation function implementation
 */

namespace MLLib {
namespace layer {
namespace activation {

/**
 * @class GELU
 * @brief Gaussian Error Linear Unit activation function
 * 
 * GELU is used in many state-of-the-art models like BERT and GPT.
 * Approximation: f(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
 */
class GELU : public Activation {
public:
  /**
   * @brief Constructor
   * @param approximate Whether to use the approximate version (default: true)
   */
  explicit GELU(bool approximate = true);

  /**
   * @brief Destructor
   */
  virtual ~GELU() = default;

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
   * @brief Get approximate flag
   * @return Whether using approximate version
   */
  bool is_approximate() const { return approximate_; }

private:
  bool approximate_;  ///< Whether to use approximate computation
};

}  // namespace activation
}  // namespace layer
}  // namespace MLLib
