#pragma once

#include "base.hpp"

/**
 * @file mse.hpp
 * @brief Mean Squared Error loss function implementation
 */

namespace MLLib {
namespace loss {

/**
 * @class MSELoss
 * @brief Mean Squared Error loss function
 */
class MSELoss : public BaseLoss {
public:
  /**
   * @brief Constructor
   */
  MSELoss() = default;

  /**
   * @brief Destructor
   */
  virtual ~MSELoss() = default;

  /**
   * @brief Compute MSE loss
   * @param predictions Predicted values [batch_size, output_size]
   * @param targets Target values [batch_size, output_size]
   * @return Loss value (scalar)
   */
  double compute_loss(const NDArray& predictions,
                      const NDArray& targets) override;

  /**
   * @brief Compute gradient of MSE loss
   * @param predictions Predicted values [batch_size, output_size]
   * @param targets Target values [batch_size, output_size]
   * @return Gradient [batch_size, output_size]
   */
  NDArray compute_gradient(const NDArray& predictions,
                           const NDArray& targets) override;
};

}  // namespace loss
}  // namespace MLLib
