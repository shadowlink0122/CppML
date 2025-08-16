#pragma once

#include "base.hpp"

/**
 * @file rmsprop.hpp
 * @brief RMSprop optimizer implementation
 */

namespace MLLib {
namespace optimizer {

/**
 * @class RMSprop
 * @brief Root Mean Square Propagation optimizer
 * 
 * RMSprop adapts the learning rate for each parameter by dividing by a running
 * average of the magnitudes of recent gradients for that parameter.
 */
class RMSprop : public BaseOptimizer {
public:
  /**
   * @brief Constructor
   * @param learning_rate Learning rate (default: 0.001)
   * @param rho Exponential decay rate (default: 0.9)
   * @param epsilon Small constant for numerical stability (default: 1e-8)
   */
  explicit RMSprop(double learning_rate = 0.001,
                   double rho = 0.9,
                   double epsilon = 1e-8);

  /**
   * @brief Destructor
   */
  virtual ~RMSprop() = default;

  /**
   * @brief Update parameters using RMSprop
   * @param parameters Vector of parameter pointers
   * @param gradients Vector of corresponding gradients
   */
  void update(const std::vector<NDArray*>& parameters,
              const std::vector<NDArray*>& gradients) override;

  /**
   * @brief Reset optimizer state
   */
  void reset();

private:
  double rho_;                ///< Exponential decay rate
  double epsilon_;            ///< Numerical stability constant
  
  std::vector<NDArray> v_;    ///< Exponential moving average of squared gradients
  bool v_initialized_;        ///< Flag for lazy initialization
};

}  // namespace optimizer
}  // namespace MLLib
