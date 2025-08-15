#pragma once

#include "../ndarray.hpp"

/**
 * @file base.hpp
 * @brief Base class for all neural network layers
 */

namespace MLLib {
namespace layer {

/**
 * @class BaseLayer
 * @brief Abstract base class for all neural network layers
 */
class BaseLayer {
public:
  /**
   * @brief Virtual destructor
   */
  virtual ~BaseLayer() = default;

  /**
   * @brief Forward propagation
   * @param input Input data
   * @return Output data
   */
  virtual NDArray forward(const NDArray& input) = 0;

  /**
   * @brief Backward propagation
   * @param grad_output Gradient from the next layer
   * @return Gradient with respect to input
   */
  virtual NDArray backward(const NDArray& grad_output) = 0;

  /**
   * @brief Get trainable parameters
   * @return Vector of parameter pointers
   */
  virtual std::vector<NDArray*> get_parameters() = 0;

  /**
   * @brief Set training mode
   * @param training True for training mode, false for inference
   */
  virtual void set_training(bool training) { is_training_ = training; }

  /**
   * @brief Check if layer is in training mode
   * @return True if in training mode
   */
  bool is_training() const { return is_training_; }

protected:
  bool is_training_ = true;
};

}  // namespace layer
}  // namespace MLLib
