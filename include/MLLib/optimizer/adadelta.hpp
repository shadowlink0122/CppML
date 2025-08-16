#pragma once

#include "base.hpp"

/**
 * @file adadelta.hpp
 * @brief AdaDelta optimizer implementation
 */

namespace MLLib {
namespace optimizer {

/**
 * @class AdaDelta
 * @brief Adaptive Delta optimizer
 *
 * AdaDelta is an extension of AdaGrad that seeks to reduce its aggressive,
 * monotonically decreasing learning rate by restricting the window of
 * accumulated past gradients.
 */
class AdaDelta : public BaseOptimizer {
public:
  /**
   * @brief Constructor
   * @param learning_rate Learning rate (default: 1.0)
   * @param rho Exponential decay rate (default: 0.9)
   * @param epsilon Small constant for numerical stability (default: 1e-6)
   */
  explicit AdaDelta(double learning_rate = 1.0, double rho = 0.9,
                    double epsilon = 1e-6);

  /**
   * @brief Destructor
   */
  virtual ~AdaDelta() = default;

  /**
   * @brief Update parameters using AdaDelta
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
  double rho_;      ///< Exponential decay rate
  double epsilon_;  ///< Numerical stability constant

  std::vector<NDArray>
      E_g2_;  ///< Exponential moving average of squared gradients
  std::vector<NDArray>
      E_dx2_;  ///< Exponential moving average of squared parameter updates
  bool state_initialized_;  ///< Flag for lazy initialization
};

}  // namespace optimizer
}  // namespace MLLib
