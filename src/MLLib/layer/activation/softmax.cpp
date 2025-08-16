#include "../../../../include/MLLib/layer/activation/softmax.hpp"
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <limits>

namespace MLLib {
namespace layer {
namespace activation {

Softmax::Softmax(int axis) : axis_(axis) {}

NDArray Softmax::forward(const NDArray& input) {
  last_input_ = input;
  forward_called_ = true;

  NDArray output(input.shape());
  
  // For now, implement for 2D arrays (batch_size, features)
  // TODO: Extend to support arbitrary dimensions and axis
  if (input.shape().size() != 2) {
    throw std::invalid_argument("Softmax currently supports only 2D arrays");
  }
  
  size_t batch_size = input.shape()[0];
  size_t features = input.shape()[1];
  
  const double* input_data = input.data();
  double* output_data = output.data();

  for (size_t batch = 0; batch < batch_size; ++batch) {
    size_t batch_offset = batch * features;
    
    // Find maximum value for numerical stability (log-sum-exp trick)
    double max_val = -std::numeric_limits<double>::infinity();
    for (size_t j = 0; j < features; ++j) {
      max_val = std::max(max_val, input_data[batch_offset + j]);
    }
    
    // Compute exp(x - max) and sum
    double sum_exp = 0.0;
    for (size_t j = 0; j < features; ++j) {
      double exp_val = std::exp(input_data[batch_offset + j] - max_val);
      output_data[batch_offset + j] = exp_val;
      sum_exp += exp_val;
    }
    
    // Normalize
    for (size_t j = 0; j < features; ++j) {
      output_data[batch_offset + j] /= sum_exp;
    }
  }
  
  last_output_ = output;
  return output;
}

NDArray Softmax::backward(const NDArray& grad_output) {
  if (!forward_called_) {
    throw std::runtime_error("Forward must be called before backward");
  }

  if (grad_output.shape() != last_input_.shape()) {
    throw std::invalid_argument("Gradient output shape mismatch");
  }

  NDArray grad_input(grad_output.shape());
  
  if (last_input_.shape().size() != 2) {
    throw std::invalid_argument("Softmax backward currently supports only 2D arrays");
  }
  
  size_t batch_size = last_input_.shape()[0];
  size_t features = last_input_.shape()[1];
  
  const double* grad_output_data = grad_output.data();
  const double* softmax_output_data = last_output_.data();
  double* grad_input_data = grad_input.data();

  for (size_t batch = 0; batch < batch_size; ++batch) {
    size_t batch_offset = batch * features;
    
    for (size_t i = 0; i < features; ++i) {
      double grad_sum = 0.0;
      
      // Compute gradient using Jacobian of softmax
      for (size_t j = 0; j < features; ++j) {
        double jacobian_ij;
        if (i == j) {
          jacobian_ij = softmax_output_data[batch_offset + i] * 
                       (1.0 - softmax_output_data[batch_offset + i]);
        } else {
          jacobian_ij = -softmax_output_data[batch_offset + i] * 
                        softmax_output_data[batch_offset + j];
        }
        grad_sum += grad_output_data[batch_offset + j] * jacobian_ij;
      }
      
      grad_input_data[batch_offset + i] = grad_sum;
    }
  }

  return grad_input;
}

}  // namespace activation
}  // namespace layer
}  // namespace MLLib
