#include "../../../../include/MLLib/layer/activation/gelu.hpp"
#include <cmath>
#include <stdexcept>

namespace MLLib {
namespace layer {
namespace activation {

GELU::GELU(bool approximate) : approximate_(approximate) {}

NDArray GELU::forward(const NDArray& input) {
  last_input_ = input;
  forward_called_ = true;

  NDArray output(input.shape());
  const double* input_data = input.data();
  double* output_data = output.data();

  if (approximate_) {
    // Approximate GELU: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    const double sqrt_2_over_pi = std::sqrt(2.0 / M_PI);
    
    for (size_t i = 0; i < input.size(); ++i) {
      double x = input_data[i];
      double x_cubed = x * x * x;
      double inner = sqrt_2_over_pi * (x + 0.044715 * x_cubed);
      output_data[i] = 0.5 * x * (1.0 + std::tanh(inner));
    }
  } else {
    // Exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    const double sqrt_2 = std::sqrt(2.0);
    
    for (size_t i = 0; i < input.size(); ++i) {
      double x = input_data[i];
      output_data[i] = 0.5 * x * (1.0 + std::erf(x / sqrt_2));
    }
  }

  return output;
}

NDArray GELU::backward(const NDArray& grad_output) {
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

  if (approximate_) {
    // Derivative of approximate GELU
    const double sqrt_2_over_pi = std::sqrt(2.0 / M_PI);
    
    for (size_t i = 0; i < grad_output.size(); ++i) {
      double x = input_data[i];
      double x_squared = x * x;
      double x_cubed = x_squared * x;
      
      double inner = sqrt_2_over_pi * (x + 0.044715 * x_cubed);
      double tanh_inner = std::tanh(inner);
      double sech_squared = 1.0 - tanh_inner * tanh_inner;
      
      double derivative = 0.5 * (1.0 + tanh_inner) + 
                         0.5 * x * sech_squared * sqrt_2_over_pi * (1.0 + 0.134145 * x_squared);
      
      grad_input_data[i] = grad_output_data[i] * derivative;
    }
  } else {
    // Derivative of exact GELU
    const double sqrt_2_over_pi = std::sqrt(2.0 / M_PI);
    const double sqrt_2 = std::sqrt(2.0);
    
    for (size_t i = 0; i < grad_output.size(); ++i) {
      double x = input_data[i];
      double erf_term = std::erf(x / sqrt_2);
      double exp_term = std::exp(-0.5 * x * x);
      
      double derivative = 0.5 * (1.0 + erf_term) + x * sqrt_2_over_pi * 0.5 * exp_term;
      grad_input_data[i] = grad_output_data[i] * derivative;
    }
  }

  return grad_input;
}

}  // namespace activation
}  // namespace layer
}  // namespace MLLib
