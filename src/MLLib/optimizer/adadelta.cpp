#include "../../../include/MLLib/optimizer/adadelta.hpp"
#include <cmath>
#include <stdexcept>

namespace MLLib {
namespace optimizer {

AdaDelta::AdaDelta(double learning_rate, double rho, double epsilon)
    : BaseOptimizer(learning_rate),
      rho_(rho),
      epsilon_(epsilon),
      state_initialized_(false) {
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

void AdaDelta::update(const std::vector<NDArray*>& parameters,
                      const std::vector<NDArray*>& gradients) {
  if (parameters.size() != gradients.size()) {
    throw std::invalid_argument("Parameters and gradients size mismatch");
  }

  // Lazy initialization of state variables
  if (!state_initialized_) {
    E_g2_.resize(parameters.size());
    E_dx2_.resize(parameters.size());
    for (size_t i = 0; i < parameters.size(); ++i) {
      E_g2_[i] = NDArray(parameters[i]->shape());
      E_dx2_[i] = NDArray(parameters[i]->shape());
      // Initialize with zeros (NDArray constructor already does this)
    }
    state_initialized_ = true;
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
    double* E_g2_data = E_g2_[i].data();
    double* E_dx2_data = E_dx2_[i].data();
    double* param_data = param.data();

    for (size_t j = 0; j < param.size(); ++j) {
      // Update exponential moving average of squared gradients
      E_g2_data[j] = rho_ * E_g2_data[j] + (1.0 - rho_) * grad_data[j] * grad_data[j];
      
      // Compute parameter update
      double rms_dx = std::sqrt(E_dx2_data[j] + epsilon_);
      double rms_g = std::sqrt(E_g2_data[j] + epsilon_);
      double dx = -(rms_dx / rms_g) * grad_data[j];
      
      // Update exponential moving average of squared parameter updates
      E_dx2_data[j] = rho_ * E_dx2_data[j] + (1.0 - rho_) * dx * dx;
      
      // Update parameter
      param_data[j] += learning_rate_ * dx;
    }
  }
}

void AdaDelta::reset() {
  state_initialized_ = false;
  E_g2_.clear();
  E_dx2_.clear();
}

}  // namespace optimizer
}  // namespace MLLib
