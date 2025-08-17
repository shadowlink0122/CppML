#pragma once

/**
 * @file MLLib.hpp
 * @brief MLLib - C++ Machine Learning Library
 * @version 1.0.0
 * @author MLLib Team
 * @date 2025-08-14
 *
 * Main header file that includes all MLLib components.
 * Include this file to access all MLLib functionality.
 */

// Core components
#include "MLLib/config.hpp"
#include "MLLib/device/device.hpp"
#include "MLLib/ndarray.hpp"

// Backend components
#include "MLLib/backend/backend.hpp"

// Data processing
#include "MLLib/data/batch.hpp"
#include "MLLib/data/loader.hpp"
#include "MLLib/data/preprocess.hpp"

// Layer base and implementations
#include "MLLib/layer/base.hpp"
#include "MLLib/layer/convolution2d.hpp"
#include "MLLib/layer/dense.hpp"
#include "MLLib/layer/dropout.hpp"
#include "MLLib/layer/flatten.hpp"
#include "MLLib/layer/pooling.hpp"

// Activation functions
#include "MLLib/layer/activation/activation.hpp"
#include "MLLib/layer/activation/relu.hpp"
#include "MLLib/layer/activation/sigmoid.hpp"
#include "MLLib/layer/activation/tanh.hpp"

// Loss functions
#include "MLLib/loss/base.hpp"
#include "MLLib/loss/cross_entropy.hpp"
#include "MLLib/loss/mse.hpp"

// Model components
#include "MLLib/model/custom.hpp"
#include "MLLib/model/functional.hpp"
#include "MLLib/model/model_io.hpp"
#include "MLLib/model/sequential.hpp"

// Autoencoder models
#include "MLLib/model/autoencoder/base.hpp"
#include "MLLib/model/autoencoder/dense.hpp"
#include "MLLib/model/autoencoder/denoising.hpp"
#include "MLLib/model/autoencoder/variational.hpp"
#include "MLLib/model/autoencoder/anomaly_detector.hpp"

// Optimization
#include "MLLib/optimizer/adam.hpp"
#include "MLLib/optimizer/base.hpp"
#include "MLLib/optimizer/sgd.hpp"

// Utility components
#include "MLLib/util/io/io.hpp"
#include "MLLib/util/io/logger.hpp"
#include "MLLib/util/misc/cast.hpp"
#include "MLLib/util/misc/error.hpp"
#include "MLLib/util/misc/random.hpp"
#include "MLLib/util/misc/settings.hpp"
#include "MLLib/util/misc/test_util.hpp"
#include "MLLib/util/number/number_util.hpp"
#include "MLLib/util/number/stats.hpp"
#include "MLLib/util/string/path.hpp"
#include "MLLib/util/string/string_util.hpp"
#include "MLLib/util/system/memory.hpp"
#include "MLLib/util/system/thread.hpp"
#include "MLLib/util/time/profiler.hpp"
#include "MLLib/util/time/timer.hpp"

/**
 * @namespace MLLib
 * @brief Main namespace for the MLLib machine learning library
 *
 * This namespace contains all classes and functions for:
 * - Neural network layers and models
 * - Data preprocessing and loading
 * - Loss functions and optimizers
 * - Activation functions and metrics
 * - Utility functions
 */
namespace MLLib {
// Library version information
constexpr const char* VERSION = "1.0.0";
constexpr int VERSION_MAJOR = 1;
constexpr int VERSION_MINOR = 0;
constexpr int VERSION_PATCH = 0;

/**
 * @brief Get the library version string
 * @return Version string in format "major.minor.patch"
 */
inline const char* getVersion() {
  return VERSION;
}

/**
 * @brief Initialize the MLLib library
 * This function should be called before using any MLLib functionality
 */
inline void initialize() {
  // Initialize random seed, GPU context, etc.
  // Implementation details would go here
}

/**
 * @brief Cleanup MLLib resources
 * This function should be called when done using MLLib
 */
inline void cleanup() {
  // Cleanup GPU context, memory pools, etc.
  // Implementation details would go here
}
}  // namespace MLLib
