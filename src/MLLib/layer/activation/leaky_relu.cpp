#include "../../../../include/MLLib/layer/activation/leaky_relu.hpp"
#include <stdexcept>

namespace MLLib {
namespace layer {
namespace activation {

LeakyReLU::LeakyReLU(double alpha) : alpha_(alpha) {
  if (alpha < 0.0) {
    throw std::invalid_argument("Alpha must be non-negative");
  }
}

NDArray LeakyReLU::forward(const NDArray& input) {
  last_input_ = input;
  forward_called_ = true;

  NDArray output(input.shape());
  const double* input_data = input.data();
  double* output_data = output.data();

  for (size_t i = 0; i < input.size(); ++i) {
    if (input_data[i] > 0.0) {
      output_data[i] = input_data[i];
    } else {
      output_data[i] = alpha_ * input_data[i];
    }
  }

  return output;
}

NDArray LeakyReLU::backward(const NDArray& grad_output) {
  if (!forward_called_) {
    throw std::runtime_error("Forward must be called before backward");
  }

  if (grad_output.shape() != last_input_.shape()) {
    throw std::invalid_argument("Gradient output shape mismatch");
  }

  NDArray grad_input(grad_output.shape());
  const double* grad_output_data = grad_output.data();
  const double* input_data = last_input_.data();
  double* grad_input_data = grad_input.data();

  for (size_t i = 0; i < grad_output.size(); ++i) {
    if (input_data[i] > 0.0) {
      grad_input_data[i] = grad_output_data[i];
    } else {
      grad_input_data[i] = alpha_ * grad_output_data[i];
    }
  }

  return grad_input;
}

}  // namespace activation
}  // namespace layer
}  // namespace MLLib
