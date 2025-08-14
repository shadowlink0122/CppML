#include "../../../../include/MLLib/layer/activation/relu.hpp"
#include <algorithm>
#include <stdexcept>

namespace MLLib {
namespace layer {
namespace activation {

NDArray ReLU::forward(const NDArray& input) {
  // Cache input for backward pass
  last_input_ = input;
  forward_called_ = true;

  NDArray output(input.shape());

  for (size_t i = 0; i < input.size(); ++i) {
    output[i] = std::max(0.0, input[i]);
  }

  return output;
}

NDArray ReLU::backward(const NDArray& grad_output) {
  if (!forward_called_) {
    throw std::runtime_error("backward() called without forward()");
  }

  if (grad_output.shape() != last_input_.shape()) {
    throw std::invalid_argument("Gradient output shape must match input shape");
  }

  NDArray grad_input(grad_output.shape());

  for (size_t i = 0; i < grad_output.size(); ++i) {
    // Derivative of ReLU: 1 if input > 0, 0 otherwise
    grad_input[i] = (last_input_[i] > 0.0) ? grad_output[i] : 0.0;
  }

  return grad_input;
}

}  // namespace activation
}  // namespace layer
}  // namespace MLLib
