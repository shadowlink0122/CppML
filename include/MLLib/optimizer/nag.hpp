#pragma once

#include "base.hpp"

/**
 * @file nag.hpp
 * @brief Nesterov Accelerated Gradient optimizer implementation
 */

namespace MLLib {
namespace optimizer {

/**
 * @class NAG
 * @brief Nesterov Accelerated Gradient optimizer
 *
 * NAG is an improvement over momentum SGD that "looks ahead" by calculating
 * the gradient at the projected position rather than the current position.
 */
class NAG : public BaseOptimizer {
public:
  /**
   * @brief Constructor
   * @param learning_rate Learning rate
   * @param momentum Momentum factor (default: 0.9)
   */
  explicit NAG(double learning_rate, double momentum = 0.9);

  /**
   * @brief Destructor
   */
  virtual ~NAG() = default;

  /**
   * @brief Update parameters using NAG
   * @param parameters Vector of parameter pointers
   * @param gradients Vector of corresponding gradients
   */
  void update(const std::vector<NDArray*>& parameters,
              const std::vector<NDArray*>& gradients) override;

  /**
   * @brief Reset optimizer state
   */
  void reset();

  /**
   * @brief Get momentum parameter
   * @return Momentum value
   */
  double get_momentum() const { return momentum_; }

private:
  double momentum_;                ///< Momentum factor
  std::vector<NDArray> velocity_;  ///< Momentum velocity for each parameter
  bool velocity_initialized_;      ///< Flag for lazy initialization
};

}  // namespace optimizer
}  // namespace MLLib
