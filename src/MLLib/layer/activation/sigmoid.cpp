#include "../../../../include/MLLib/layer/activation/sigmoid.hpp"
#include <cmath>

namespace MLLib {
namespace layer {
namespace activation {

NDArray Sigmoid::forward(const NDArray& input) {
  // Cache input for backward pass
  last_input_ = input;

  NDArray output(input.shape());

  for (size_t i = 0; i < input.size(); ++i) {
    // Sigmoid: 1 / (1 + exp(-x))
    output[i] = 1.0 / (1.0 + std::exp(-input[i]));
  }

  // Cache output for backward pass
  last_output_ = output;

  return output;
}

NDArray Sigmoid::backward(const NDArray& grad_output) {
  NDArray grad_input(grad_output.shape());

  for (size_t i = 0; i < grad_output.size(); ++i) {
    // Derivative of sigmoid: sigmoid(x) * (1 - sigmoid(x))
    double sigmoid_val = last_output_[i];
    grad_input[i] = grad_output[i] * sigmoid_val * (1.0 - sigmoid_val);
  }

  return grad_input;
}

}  // namespace activation
}  // namespace layer
}  // namespace MLLib
