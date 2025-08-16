#include "../../../include/MLLib/optimizer/adam.hpp"
#include <cmath>
#include <stdexcept>

namespace MLLib {
namespace optimizer {

Adam::Adam(double learning_rate, double beta1, double beta2, double epsilon)
    : BaseOptimizer(learning_rate), beta1_(beta1), beta2_(beta2),
      epsilon_(epsilon), timestep_(0), moments_initialized_(false) {
  if (learning_rate <= 0.0) {
    throw std::invalid_argument("Learning rate must be positive");
  }
  if (beta1 < 0.0 || beta1 >= 1.0) {
    throw std::invalid_argument("beta1 must be in [0, 1)");
  }
  if (beta2 < 0.0 || beta2 >= 1.0) {
    throw std::invalid_argument("beta2 must be in [0, 1)");
  }
  if (epsilon <= 0.0) {
    throw std::invalid_argument("epsilon must be positive");
  }
}

void Adam::update(const std::vector<NDArray*>& parameters,
                  const std::vector<NDArray*>& gradients) {
  if (parameters.size() != gradients.size()) {
    throw std::invalid_argument("Parameters and gradients size mismatch");
  }

  // Lazy initialization of moment estimates
  if (!moments_initialized_) {
    m_.resize(parameters.size());
    v_.resize(parameters.size());
    for (size_t i = 0; i < parameters.size(); ++i) {
      m_[i] = NDArray(parameters[i]->shape());
      v_[i] = NDArray(parameters[i]->shape());
      // Initialize with zeros (NDArray constructor already does this)
    }
    moments_initialized_ = true;
  }

  timestep_++;

  // Bias correction factors
  double bias_correction1 = 1.0 - std::pow(beta1_, timestep_);
  double bias_correction2 = 1.0 - std::pow(beta2_, timestep_);

  for (size_t i = 0; i < parameters.size(); ++i) {
    if (parameters[i] == nullptr || gradients[i] == nullptr) {
      throw std::invalid_argument("Null parameter or gradient pointer");
    }

    const NDArray& grad = *gradients[i];
    NDArray& param = *parameters[i];

    if (param.shape() != grad.shape()) {
      throw std::invalid_argument("Parameter and gradient shape mismatch");
    }

    // Update biased first moment estimate: m = beta1 * m + (1 - beta1) * grad
    const double* grad_data = grad.data();
    double* m_data = m_[i].data();
    double* v_data = v_[i].data();
    double* param_data = param.data();

    for (size_t j = 0; j < param.size(); ++j) {
      // Update first moment
      m_data[j] = beta1_ * m_data[j] + (1.0 - beta1_) * grad_data[j];

      // Update second moment
      v_data[j] =
          beta2_ * v_data[j] + (1.0 - beta2_) * grad_data[j] * grad_data[j];

      // Bias-corrected first moment
      double m_hat = m_data[j] / bias_correction1;

      // Bias-corrected second moment
      double v_hat = v_data[j] / bias_correction2;

      // Update parameter
      param_data[j] -= learning_rate_ * m_hat / (std::sqrt(v_hat) + epsilon_);
    }
  }
}

void Adam::reset() {
  timestep_ = 0;
  moments_initialized_ = false;
  m_.clear();
  v_.clear();
}

}  // namespace optimizer
}  // namespace MLLib
