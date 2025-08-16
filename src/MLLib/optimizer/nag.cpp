#include "../../../include/MLLib/optimizer/nag.hpp"
#include <stdexcept>

namespace MLLib {
namespace optimizer {

NAG::NAG(double learning_rate, double momentum)
    : BaseOptimizer(learning_rate),
      momentum_(momentum),
      velocity_initialized_(false) {
  if (learning_rate <= 0.0) {
    throw std::invalid_argument("Learning rate must be positive");
  }
  if (momentum < 0.0 || momentum >= 1.0) {
    throw std::invalid_argument("Momentum must be in [0, 1)");
  }
}

void NAG::update(const std::vector<NDArray*>& parameters,
                 const std::vector<NDArray*>& gradients) {
  if (parameters.size() != gradients.size()) {
    throw std::invalid_argument("Parameters and gradients size mismatch");
  }

  // Lazy initialization of velocity
  if (!velocity_initialized_) {
    velocity_.resize(parameters.size());
    for (size_t i = 0; i < parameters.size(); ++i) {
      velocity_[i] = NDArray(parameters[i]->shape());
      // Initialize with zeros (NDArray constructor already does this)
    }
    velocity_initialized_ = true;
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
    double* velocity_data = velocity_[i].data();
    double* param_data = param.data();

    for (size_t j = 0; j < param.size(); ++j) {
      // NAG update:
      // v_t = momentum * v_{t-1} - learning_rate * gradient
      // param = param + momentum * v_t - learning_rate * gradient
      
      double old_velocity = velocity_data[j];
      velocity_data[j] = momentum_ * velocity_data[j] - learning_rate_ * grad_data[j];
      
      // Nesterov update: param = param + momentum * (v_t - v_{t-1}) + v_t
      // Simplified: param = param + momentum * v_t - momentum * v_{t-1} + v_t
      //           = param + momentum * v_t + v_t - momentum * v_{t-1}
      //           = param + (1 + momentum) * v_t - momentum * v_{t-1}
      param_data[j] += (1.0 + momentum_) * velocity_data[j] - momentum_ * old_velocity;
    }
  }
}

void NAG::reset() {
  velocity_initialized_ = false;
  velocity_.clear();
}

}  // namespace optimizer
}  // namespace MLLib
