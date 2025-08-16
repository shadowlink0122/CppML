#include "../../../include/MLLib/optimizer/adagrad.hpp"
#include <cmath>
#include <stdexcept>

namespace MLLib {
namespace optimizer {

AdaGrad::AdaGrad(double learning_rate, double epsilon)
    : BaseOptimizer(learning_rate), epsilon_(epsilon), G_initialized_(false) {
  if (learning_rate <= 0.0) {
    throw std::invalid_argument("Learning rate must be positive");
  }
  if (epsilon <= 0.0) {
    throw std::invalid_argument("epsilon must be positive");
  }
}

void AdaGrad::update(const std::vector<NDArray*>& parameters,
                     const std::vector<NDArray*>& gradients) {
  if (parameters.size() != gradients.size()) {
    throw std::invalid_argument("Parameters and gradients size mismatch");
  }

  // Lazy initialization of accumulated squared gradients
  if (!G_initialized_) {
    G_.resize(parameters.size());
    for (size_t i = 0; i < parameters.size(); ++i) {
      G_[i] = NDArray(parameters[i]->shape());
      // Initialize with zeros (NDArray constructor already does this)
    }
    G_initialized_ = true;
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
    double* G_data = G_[i].data();
    double* param_data = param.data();

    for (size_t j = 0; j < param.size(); ++j) {
      // Accumulate squared gradients
      G_data[j] += grad_data[j] * grad_data[j];

      // Update parameter
      param_data[j] -=
          learning_rate_ * grad_data[j] / (std::sqrt(G_data[j]) + epsilon_);
    }
  }
}

void AdaGrad::reset() {
  G_initialized_ = false;
  G_.clear();
}

}  // namespace optimizer
}  // namespace MLLib
