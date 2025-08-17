#include "MLLib/layer/dense.hpp"
#include "MLLib/backend/backend.hpp"
#include <cmath>
#include <random>

using namespace std;

namespace MLLib {
namespace layer {

Dense::Dense(size_t input_size, size_t output_size, bool use_bias)
    : input_size_(input_size), output_size_(output_size), use_bias_(use_bias) {
  initialize_parameters();
}

NDArray Dense::forward(const NDArray& input) {
  // Cache input for backward pass
  last_input_ = input;

  // Input shape: [batch_size, input_size]
  // Weights shape: [input_size, output_size]
  // Output shape: [batch_size, output_size]

  NDArray output = input.matmul(weights_);

  if (use_bias_) {
    // Add bias to each sample in the batch
    const auto& shape = output.shape();
    size_t batch_size = shape[0];
    size_t output_size = shape[1];

    for (size_t i = 0; i < batch_size; ++i) {
      for (size_t j = 0; j < output_size; ++j) {
        output.at({i, j}) += bias_[j];
      }
    }
  }

  return output;
}

NDArray Dense::backward(const NDArray& grad_output) {
  // grad_output shape: [batch_size, output_size]
  // weights shape: [input_size, output_size]
  // last_input shape: [batch_size, input_size]

  // Compute gradient w.r.t. weights: input^T * grad_output
  // last_input^T shape: [input_size, batch_size]
  // grad_output shape: [batch_size, output_size]
  // weight_gradients shape: [input_size, output_size]

  // Transpose last_input
  const auto& input_shape = last_input_.shape();
  size_t batch_size = input_shape[0];
  size_t input_size = input_shape[1];

  NDArray input_transposed({input_size, batch_size});
  for (size_t i = 0; i < input_size; ++i) {
    for (size_t j = 0; j < batch_size; ++j) {
      input_transposed.at({i, j}) = last_input_.at({j, i});
    }
  }

  weight_gradients_ = input_transposed.matmul(grad_output);

  // Compute gradient w.r.t. bias: sum over batch dimension
  if (use_bias_) {
    const auto& grad_shape = grad_output.shape();
    size_t batch_size = grad_shape[0];
    size_t output_size = grad_shape[1];

    bias_gradients_ = NDArray({output_size});
    for (size_t j = 0; j < output_size; ++j) {
      double sum = 0.0;
      for (size_t i = 0; i < batch_size; ++i) {
        sum += grad_output.at({i, j});
      }
      bias_gradients_[j] = sum;
    }
  }

  // Compute gradient w.r.t. input: grad_output * weights^T
  // grad_output shape: [batch_size, output_size]
  // weights^T shape: [output_size, input_size]
  // grad_input shape: [batch_size, input_size]

  // Transpose weights
  NDArray weights_transposed({output_size_, input_size_});
  for (size_t i = 0; i < output_size_; ++i) {
    for (size_t j = 0; j < input_size_; ++j) {
      weights_transposed.at({i, j}) = weights_.at({j, i});
    }
  }

  NDArray grad_input = grad_output.matmul(weights_transposed);

  return grad_input;
}

std::vector<NDArray*> Dense::get_parameters() {
  std::vector<NDArray*> params;
  params.push_back(&weights_);
  if (use_bias_) {
    params.push_back(&bias_);
  }
  return params;
}

void Dense::initialize_parameters() {
  // Xavier/Glorot initialization
  std::random_device rd;
  std::mt19937 gen(rd());

  double limit = std::sqrt(6.0 / (input_size_ + output_size_));
  std::uniform_real_distribution<double> dis(-limit, limit);

  // Initialize weights
  weights_ = NDArray({input_size_, output_size_});
  for (size_t i = 0; i < input_size_; ++i) {
    for (size_t j = 0; j < output_size_; ++j) {
      weights_.at({i, j}) = dis(gen);
    }
  }

  // Initialize gradients
  weight_gradients_ = NDArray({input_size_, output_size_});

  // Initialize bias
  if (use_bias_) {
    bias_ = NDArray({output_size_});
    bias_.fill(0.0);
    bias_gradients_ = NDArray({output_size_});
  }
}

}  // namespace layer
}  // namespace MLLib
