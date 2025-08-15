#include "../../../include/MLLib/loss/mse.hpp"
#include <cmath>
#include <stdexcept>

namespace MLLib {
namespace loss {

double MSELoss::compute_loss(const NDArray& predictions,
                             const NDArray& targets) {
  if (predictions.shape() != targets.shape()) {
    throw std::invalid_argument(
        "Predictions and targets must have the same shape");
  }

  double total_loss = 0.0;
  size_t total_elements = predictions.size();

  for (size_t i = 0; i < total_elements; ++i) {
    double diff = predictions[i] - targets[i];
    total_loss += diff * diff;
  }

  // Return mean squared error
  return total_loss / total_elements;
}

NDArray MSELoss::compute_gradient(const NDArray& predictions,
                                  const NDArray& targets) {
  if (predictions.shape() != targets.shape()) {
    throw std::invalid_argument(
        "Predictions and targets must have the same shape");
  }

  NDArray gradient(predictions.shape());
  size_t total_elements = predictions.size();

  for (size_t i = 0; i < total_elements; ++i) {
    // Gradient of MSE: 2 * (predictions - targets) / n
    gradient[i] = 2.0 * (predictions[i] - targets[i]) / total_elements;
  }

  return gradient;
}

}  // namespace loss
}  // namespace MLLib
