/**
 * @file gpu_kernel_manager_stub.cpp
 * @brief CPU-based stub implementation for GPU kernel management
 * Used in CI environments where GPU/Metal support is not available
 */

#include "../../../include/MLLib/backend/gpu_kernel_manager.hpp"
#include <chrono>
#include <cmath>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace MLLib {
namespace Backend {

// Static member initialization
bool GPUKernelManager::initialized_ = false;
std::unordered_map<std::string, ActivationKernelRegistry::ActivationDef>
    ActivationKernelRegistry::activations_;

// GPUKernelManager Implementation (CPU fallback)
void GPUKernelManager::executeUnaryKernel(const std::string& name,
                                          const double* input, double* output,
                                          size_t size,
                                          const std::vector<double>& params) {

  std::cout << "⚠️  Using CPU fallback for GPU kernel: " << name << std::endl;

  // Simple CPU implementations for common operations
  if (name == "relu") {
    for (size_t i = 0; i < size; ++i) {
      output[i] = std::max(0.0, input[i]);
    }
  } else if (name == "sigmoid") {
    for (size_t i = 0; i < size; ++i) {
      output[i] = 1.0 / (1.0 + std::exp(-input[i]));
    }
  } else if (name == "tanh") {
    for (size_t i = 0; i < size; ++i) {
      output[i] = std::tanh(input[i]);
    }
  } else if (name == "leaky_relu") {
    double alpha = params.empty() ? 0.01 : params[0];
    for (size_t i = 0; i < size; ++i) {
      output[i] = input[i] > 0 ? input[i] : alpha * input[i];
    }
  } else if (name == "elu") {
    double alpha = params.empty() ? 1.0 : params[0];
    for (size_t i = 0; i < size; ++i) {
      output[i] = input[i] > 0 ? input[i] : alpha * (std::exp(input[i]) - 1.0);
    }
  } else if (name == "softplus") {
    for (size_t i = 0; i < size; ++i) {
      output[i] = std::log(1.0 + std::exp(input[i]));
    }
  } else {
    // Default: copy input to output for unknown kernels
    std::cout << "⚠️  Unknown kernel: " << name << ", using identity"
              << std::endl;
    for (size_t i = 0; i < size; ++i) {
      output[i] = input[i];
    }
  }
}

void GPUKernelManager::executeBinaryKernel(
    const std::string& name, const double* input1, const double* input2,
    double* output, size_t size, const std::vector<double>& /* params */) {

  std::cout << "⚠️  Using CPU fallback for binary GPU kernel: " << name
            << std::endl;

  if (name == "add") {
    for (size_t i = 0; i < size; ++i) {
      output[i] = input1[i] + input2[i];
    }
  } else if (name == "multiply") {
    for (size_t i = 0; i < size; ++i) {
      output[i] = input1[i] * input2[i];
    }
  } else if (name == "subtract") {
    for (size_t i = 0; i < size; ++i) {
      output[i] = input1[i] - input2[i];
    }
  } else {
    // Default: copy first input to output
    for (size_t i = 0; i < size; ++i) {
      output[i] = input1[i];
    }
  }
}

void GPUKernelManager::initializeBuiltinKernels() {
  if (initialized_) {
    return;
  }

  std::cout << "🔧 Initializing GPU kernel manager (CPU fallback mode)"
            << std::endl;

  // Initialize activation kernel registry
  ActivationKernelRegistry::initializeBuiltinActivations();

  initialized_ = true;
  std::cout << "✅ GPU kernel manager initialized (CPU fallback)" << std::endl;
}

void GPUKernelManager::cleanup() {
  if (!initialized_) {
    return;
  }

  std::cout << "🧹 Cleaning up GPU kernel manager (CPU fallback)" << std::endl;
  initialized_ = false;
}

// ActivationKernelRegistry Implementation (CPU fallback)
void ActivationKernelRegistry::executeActivation(
    const std::string& name, const double* input, double* output, size_t size,
    const std::vector<double>& params) {

  auto it = activations_.find(name);
  if (it != activations_.end()) {
    std::cout << "🔧 Executing activation " << name << " (CPU fallback)"
              << std::endl;

    // Use CPU implementation via kernel manager
    GPUKernelManager::executeUnaryKernel(name, input, output, size, params);
  } else {
    std::cout << "⚠️  Unknown activation: " << name << ", using identity"
              << std::endl;
    for (size_t i = 0; i < size; ++i) {
      output[i] = input[i];
    }
  }
}

void ActivationKernelRegistry::registerActivation(const ActivationDef& def) {
  std::cout << "📝 Registering activation: " << def.name << " (CPU fallback)"
            << std::endl;
  activations_[def.name] = def;
}

void ActivationKernelRegistry::initializeBuiltinActivations() {
  std::cout << "🔧 Initializing builtin activations (CPU fallback mode)"
            << std::endl;

  // Register common activation functions
  ActivationDef relu = {"relu", "max(0.0f, input)", {}, false};
  activations_["relu"] = relu;

  ActivationDef sigmoid = {"sigmoid", "1.0f / (1.0f + exp(-input))", {}, false};
  activations_["sigmoid"] = sigmoid;

  ActivationDef tanh = {"tanh", "tanh(input)", {}, false};
  activations_["tanh"] = tanh;

  ActivationDef leaky_relu = {"leaky_relu",
                              "input > 0.0f ? input : alpha * input",
                              {"alpha"},
                              true};
  activations_["leaky_relu"] = leaky_relu;

  ActivationDef elu = {"elu",
                       "input > 0.0f ? input : alpha * (exp(input) - 1.0f)",
                       {"alpha"},
                       true};
  activations_["elu"] = elu;

  ActivationDef softplus = {"softplus", "log(1.0f + exp(input))", {}, false};
  activations_["softplus"] = softplus;

  std::cout << "✅ Builtin activations initialized (" << activations_.size()
            << " activations)" << std::endl;
}

}  // namespace Backend
}  // namespace MLLib
