#include "../../../include/MLLib/optimizer/sgd.hpp"
#include <stdexcept>

namespace MLLib {
namespace optimizer {

SGD::SGD(double learning_rate, double momentum)
    : BaseOptimizer(learning_rate), momentum_(momentum),
      momentum_initialized_(false) {}

void SGD::update(const std::vector<NDArray*>& parameters,
                 const std::vector<NDArray*>& gradients) {
  if (parameters.size() != gradients.size()) {
    throw std::invalid_argument(
        "Number of parameters and gradients must match");
  }

  // Initialize velocity if using momentum and not yet initialized
  if (momentum_ > 0.0 && !momentum_initialized_) {
    velocity_.clear();
    velocity_.reserve(parameters.size());

    for (const auto& param : parameters) {
      velocity_.emplace_back(param->shape());
      velocity_.back().fill(0.0);
    }
    momentum_initialized_ = true;
  }

  for (size_t i = 0; i < parameters.size(); ++i) {
    NDArray* param = parameters[i];
    const NDArray* grad = gradients[i];

    if (momentum_ > 0.0) {
      // Update velocity: v = momentum * v - learning_rate * gradient
      for (size_t j = 0; j < velocity_[i].size(); ++j) {
        velocity_[i][j] =
            momentum_ * velocity_[i][j] - learning_rate_ * (*grad)[j];
      }

      // Update parameters: param = param + velocity
      for (size_t j = 0; j < param->size(); ++j) {
        (*param)[j] += velocity_[i][j];
      }
    } else {
      // Simple SGD: param = param - learning_rate * gradient
      for (size_t j = 0; j < param->size(); ++j) {
        (*param)[j] -= learning_rate_ * (*grad)[j];
      }
    }
  }
}

}  // namespace optimizer
}  // namespace MLLib
