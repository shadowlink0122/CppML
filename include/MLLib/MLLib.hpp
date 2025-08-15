#pragma once

/**
 * @file MLLib.hpp
 * @brief Main header file for MLLib - includes all necessary components
 */

// Core components
#include "config.hpp"
#include "ndarray.hpp"

// Device management
#include "device/device.hpp"

// Layers
#include "layer/activation/activation.hpp"
#include "layer/activation/relu.hpp"
#include "layer/activation/sigmoid.hpp"
#include "layer/base.hpp"
#include "layer/dense.hpp"

// Loss functions
#include "loss/base.hpp"
#include "loss/mse.hpp"

// Optimizers
#include "optimizer/base.hpp"
#include "optimizer/sgd.hpp"

// Models
#include "model/model_io.hpp"
#include "model/sequential.hpp"

// Backend operations
#include "backend/backend.hpp"

/**
 * @namespace MLLib
 * @brief Main namespace for the MLLib machine learning library
 */
namespace MLLib {

/**
 * @brief Library version information
 */
constexpr const char* VERSION = "1.0.0";

/**
 * @brief Get library version
 * @return Version string
 */
inline const char* get_version() {
  return VERSION;
}

}  // namespace MLLib
