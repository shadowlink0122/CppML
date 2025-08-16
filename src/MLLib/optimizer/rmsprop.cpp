#include "../../../include/MLLib/optimizer/rmsprop.hpp"
#include <cmath>
#include <stdexcept>

namespace MLLib {
namespace optimizer {

RMSprop::RMSprop(double learning_rate, double rho, double epsilon)
    : BaseOptimizer(learning_rate),
      rho_(rho),
      epsilon_(epsilon),
      v_initialized_(false) {
  if (learning_rate <= 0.0) {
    throw std::invalid_argument("Learning rate must be positive");
  }
  if (rho < 0.0 || rho >= 1.0) {
    throw std::invalid_argument("rho must be in [0, 1)");
  }
  if (epsilon <= 0.0) {
    throw std::invalid_argument("epsilon must be positive");
  }
}

void RMSprop::update(const std::vector<NDArray*>& parameters,
                     const std::vector<NDArray*>& gradients) {
  if (parameters.size() != gradients.size()) {
    throw std::invalid_argument("Parameters and gradients size mismatch");
  }

  // Lazy initialization of moving average
  if (!v_initialized_) {
    v_.resize(parameters.size());
    for (size_t i = 0; i < parameters.size(); ++i) {
      v_[i] = NDArray(parameters[i]->shape());
      // Initialize with zeros (NDArray constructor already does this)
    }
    v_initialized_ = true;
  }

  for (size_t i = 0; i < parameters.size(); ++i) {
    if (parameters[i] == nullptr || gradients[i] == nullptr) {
      throw std::invalid_argument("Null parameter or gradient pointer");
    }

    const NDArray& grad = *gradients[i];
    NDArray& param = *parameters[i];

    if (param.shape() != grad.shape()) {
      throw std::invalid_argument("Parameter and gradient shape mismatch");
    }

    const double* grad_data = grad.data();
    double* v_data = v_[i].data();
    double* param_data = param.data();

    for (size_t j = 0; j < param.size(); ++j) {
      // Update exponential moving average of squared gradients
      v_data[j] = rho_ * v_data[j] + (1.0 - rho_) * grad_data[j] * grad_data[j];
      
      // Update parameter
      param_data[j] -= learning_rate_ * grad_data[j] / (std::sqrt(v_data[j]) + epsilon_);
    }
  }
}

void RMSprop::reset() {
  v_initialized_ = false;
  v_.clear();
}

}  // namespace optimizer
}  // namespace MLLib
