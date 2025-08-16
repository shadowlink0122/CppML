#pragma once

#include "base.hpp"

/**
 * @file adam.hpp
 * @brief Adam optimizer implementation
 */

namespace MLLib {
namespace optimizer {

/**
 * @class Adam
 * @brief Adaptive Moment Estimation optimizer
 * 
 * Adam combines the advantages of AdaGrad and RMSprop by computing adaptive
 * learning rates for each parameter using estimates of first and second moments
 * of the gradients.
 */
class Adam : public BaseOptimizer {
public:
  /**
   * @brief Constructor
   * @param learning_rate Learning rate (default: 0.001)
   * @param beta1 Exponential decay rate for first moment estimates (default: 0.9)
   * @param beta2 Exponential decay rate for second moment estimates (default: 0.999)
   * @param epsilon Small constant for numerical stability (default: 1e-8)
   */
  explicit Adam(double learning_rate = 0.001, 
                double beta1 = 0.9,
                double beta2 = 0.999,
                double epsilon = 1e-8);

  /**
   * @brief Destructor
   */
  virtual ~Adam() = default;

  /**
   * @brief Update parameters using Adam
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
  double beta1_;              ///< First moment decay rate
  double beta2_;              ///< Second moment decay rate  
  double epsilon_;            ///< Numerical stability constant
  int timestep_;              ///< Current timestep for bias correction
  
  std::vector<NDArray> m_;    ///< First moment estimates
  std::vector<NDArray> v_;    ///< Second moment estimates
  bool moments_initialized_;  ///< Flag for lazy initialization
};

}  // namespace optimizer
}  // namespace MLLib
