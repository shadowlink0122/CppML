#pragma once

#include "../ndarray.hpp"
#include <vector>

/**
 * @file base.hpp
 * @brief Base class for optimizers
 */

namespace MLLib {
namespace optimizer {

/**
 * @class BaseOptimizer
 * @brief Abstract base class for optimizers
 */
class BaseOptimizer {
public:
  /**
   * @brief Constructor
   * @param learning_rate Learning rate
   */
  explicit BaseOptimizer(double learning_rate)
      : learning_rate_(learning_rate) {}

  /**
   * @brief Virtual destructor
   */
  virtual ~BaseOptimizer() = default;

  /**
   * @brief Update parameters
   * @param parameters Vector of parameter pointers
   * @param gradients Vector of corresponding gradients
   */
  virtual void update(const std::vector<NDArray*>& parameters,
                      const std::vector<NDArray*>& gradients) = 0;

  /**
   * @brief Get learning rate
   * @return Learning rate
   */
  double get_learning_rate() const { return learning_rate_; }

  /**
   * @brief Set learning rate
   * @param learning_rate New learning rate
   */
  void set_learning_rate(double learning_rate) {
    learning_rate_ = learning_rate;
  }

protected:
  double learning_rate_;
};

}  // namespace optimizer
}  // namespace MLLib
