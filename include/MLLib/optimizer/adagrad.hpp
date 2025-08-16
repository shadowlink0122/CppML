#pragma once

#include "base.hpp"

/**
 * @file adagrad.hpp
 * @brief AdaGrad optimizer implementation
 */

namespace MLLib {
namespace optimizer {

/**
 * @class AdaGrad
 * @brief Adaptive Gradient Algorithm optimizer
 * 
 * AdaGrad adapts the learning rate to the parameters, performing larger
 * updates for infrequent parameters and smaller updates for frequent parameters.
 */
class AdaGrad : public BaseOptimizer {
public:
  /**
   * @brief Constructor
   * @param learning_rate Learning rate (default: 0.01)
   * @param epsilon Small constant for numerical stability (default: 1e-8)
   */
  explicit AdaGrad(double learning_rate = 0.01,
                   double epsilon = 1e-8);

  /**
   * @brief Destructor
   */
  virtual ~AdaGrad() = default;

  /**
   * @brief Update parameters using AdaGrad
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
  double epsilon_;               ///< Numerical stability constant
  
  std::vector<NDArray> G_;       ///< Accumulated squared gradients
  bool G_initialized_;           ///< Flag for lazy initialization
};

}  // namespace optimizer
}  // namespace MLLib
