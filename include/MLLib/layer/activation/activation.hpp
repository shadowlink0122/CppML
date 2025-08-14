#pragma once

#include "../base.hpp"

/**
 * @file activation.hpp
 * @brief Base class for activation functions
 */

namespace MLLib {
namespace layer {
namespace activation {

/**
 * @class Activation
 * @brief Base class for activation functions
 */
class Activation : public BaseLayer {
public:
  /**
   * @brief Virtual destructor
   */
  virtual ~Activation() = default;

  /**
   * @brief Get trainable parameters (activation functions have no parameters)
   * @return Empty vector
   */
  std::vector<NDArray*> get_parameters() override { return {}; }

protected:
  NDArray last_input_;  ///< Cache input for backward pass
};

}  // namespace activation
}  // namespace layer
}  // namespace MLLib
