#pragma once

#include "base.hpp"

/**
 * @file sgd.hpp
 * @brief Stochastic Gradient Descent optimizer implementation
 */

namespace MLLib {
namespace optimizer {

/**
 * @class SGD
 * @brief Stochastic Gradient Descent optimizer
 */
class SGD : public BaseOptimizer {
public:
  /**
   * @brief Constructor
   * @param learning_rate Learning rate
   * @param momentum Momentum factor (default: 0.0)
   */
  explicit SGD(double learning_rate, double momentum = 0.0);

  /**
   * @brief Destructor
   */
  virtual ~SGD() = default;

  /**
   * @brief Update parameters using SGD
   * @param parameters Vector of parameter pointers
   * @param gradients Vector of corresponding gradients
   */
  void update(const std::vector<NDArray*>& parameters,
              const std::vector<NDArray*>& gradients) override;

private:
  double momentum_;
  std::vector<NDArray> velocity_;  ///< Momentum velocity for each parameter
  bool momentum_initialized_;
};

}  // namespace optimizer
}  // namespace MLLib
