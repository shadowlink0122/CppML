#pragma once

#include <cstddef>  // for size_t

/**
 * @file config.hpp
 * @brief Configuration and global settings for MLLib
 */

namespace MLLib {

/**
 * @namespace config
 * @brief Configuration constants and settings
 */
namespace config {

// Version information
constexpr const char* VERSION = "1.0.0";
constexpr int VERSION_MAJOR = 1;
constexpr int VERSION_MINOR = 0;
constexpr int VERSION_PATCH = 0;

// Numerical constants
constexpr double EPSILON = 1e-8;  ///< Small value to prevent division by zero

// PI constant (C++20 std::numbers::pi equivalent for C++17 compatibility)
#ifdef __cpp_lib_math_constants
#include <numbers>
constexpr double PI = std::numbers::pi;
#else
constexpr double PI = 3.14159265358979323846264338327950288;
#endif

// Default values
constexpr size_t DEFAULT_BATCH_SIZE = 32;
constexpr double DEFAULT_LEARNING_RATE = 0.001;
constexpr int DEFAULT_EPOCHS = 1000;

// Random seed
constexpr unsigned int DEFAULT_RANDOM_SEED = 42;

}  // namespace config
}  // namespace MLLib
