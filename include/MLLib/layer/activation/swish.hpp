#pragma once

#include "activation.hpp"

/**
 * @file swish.hpp
 * @brief Swish/SiLU activation function implementation
 */

namespace MLLib {
namespace layer {
namespace activation {

/**
 * @class Swish
 * @brief Swish activation function (also known as SiLU)
 *
 * Swish is a self-gated activation function: f(x) = x * sigmoid(beta * x)
 * When beta = 1, it's equivalent to SiLU (Sigmoid Linear Unit)
 */
class Swish : public Activation {
public:
  /**
   * @brief Constructor
   * @param beta Scaling parameter (default: 1.0 for SiLU)
   */
  explicit Swish(double beta = 1.0);

  /**
   * @brief Destructor
   */
  virtual ~Swish() = default;

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
   * @brief Get beta parameter
   * @return Beta value
   */
  double get_beta() const { return beta_; }

private:
  double beta_;  ///< Scaling parameter
};

}  // namespace activation
}  // namespace layer
}  // namespace MLLib
