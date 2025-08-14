#include "../../../../include/MLLib/layer/activation/tanh.hpp"
#include <cmath>
#include <stdexcept>

namespace MLLib {
namespace layer {
namespace activation {

NDArray Tanh::forward(const NDArray& input) {
  // Store input for backward pass
  last_input_ = input;
  forward_called_ = true;

  NDArray output(input.shape());

  for (size_t i = 0; i < input.size(); ++i) {
    output[i] = std::tanh(input[i]);
  }

  return output;
}

NDArray Tanh::backward(const NDArray& grad_output) {
  if (!forward_called_) {
    throw std::runtime_error("backward() called without forward()");
  }

  if (grad_output.shape() != last_input_.shape()) {
    throw std::invalid_argument("Gradient output shape must match input shape");
  }

  NDArray grad_input(last_input_.shape());

  for (size_t i = 0; i < last_input_.size(); ++i) {
    double tanh_val = std::tanh(last_input_[i]);
    // Derivative of tanh: 1 - tanh²(x)
    grad_input[i] = grad_output[i] * (1.0 - tanh_val * tanh_val);
  }

  return grad_input;
}

}  // namespace activation
}  // namespace layer
}  // namespace MLLib
