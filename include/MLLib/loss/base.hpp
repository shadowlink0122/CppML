#pragma once

#include "../ndarray.hpp"

/**
 * @file base.hpp
 * @brief Base class for loss functions
 */

namespace MLLib {
namespace loss {

/**
 * @class BaseLoss
 * @brief Abstract base class for loss functions
 */
class BaseLoss {
public:
  /**
   * @brief Virtual destructor
   */
  virtual ~BaseLoss() = default;

  /**
   * @brief Compute loss
   * @param predictions Predicted values
   * @param targets Target values
   * @return Loss value
   */
  virtual double compute_loss(const NDArray& predictions,
                              const NDArray& targets) = 0;

  /**
   * @brief Compute gradient of loss with respect to predictions
   * @param predictions Predicted values
   * @param targets Target values
   * @return Gradient
   */
  virtual NDArray compute_gradient(const NDArray& predictions,
                                   const NDArray& targets) = 0;
};

}  // namespace loss
}  // namespace MLLib
