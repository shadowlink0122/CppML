#pragma once

#include "../../../include/MLLib/config.hpp"
#include "../../common/test_utils.hpp"

namespace MLLib {
namespace test {

/**
 * @class ConfigConstantsTest
 * @brief Test config constants and values
 */
class ConfigConstantsTest : public TestCase {
public:
  ConfigConstantsTest() : TestCase("ConfigConstantsTest") {}

protected:
  void test() override {
    // Test version constants
    assertEqual(std::string("1.0.0"), std::string(config::VERSION),
                "VERSION should be 1.0.0");
    assertEqual(1, config::VERSION_MAJOR, "VERSION_MAJOR should be 1");
    assertEqual(0, config::VERSION_MINOR, "VERSION_MINOR should be 0");
    assertEqual(0, config::VERSION_PATCH, "VERSION_PATCH should be 0");

    // Test numerical constants
    assertNear(1e-8, config::EPSILON, 1e-12, "EPSILON should be 1e-8");
    assertNear(3.14159265358979323846264338327950288, config::PI, 1e-15,
               "PI should be accurate");

    // Test default values
    assertEqual(size_t(32), config::DEFAULT_BATCH_SIZE,
                "DEFAULT_BATCH_SIZE should be 32");
    assertNear(0.001, config::DEFAULT_LEARNING_RATE, 1e-9,
               "DEFAULT_LEARNING_RATE should be 0.001");
    assertEqual(1000, config::DEFAULT_EPOCHS, "DEFAULT_EPOCHS should be 1000");
    assertEqual(42u, config::DEFAULT_RANDOM_SEED,
                "DEFAULT_RANDOM_SEED should be 42");
  }
};

/**
 * @class ConfigUsageTest
 * @brief Test practical usage of config values
 */
class ConfigUsageTest : public TestCase {
public:
  ConfigUsageTest() : TestCase("ConfigUsageTest") {}

protected:
  void test() override {
    // Test epsilon usage for division by zero protection
    double small_value = config::EPSILON / 2.0;
    double result = 1.0 / (small_value + config::EPSILON);
    assertTrue(std::isfinite(result),
               "Epsilon should prevent division by zero");
    assertTrue(result > 0, "Result should be positive");

    // Test PI usage for trigonometric calculations
    double cos_pi = std::cos(config::PI);
    assertNear(-1.0, cos_pi, 1e-10, "cos(PI) should be -1");

    double sin_half_pi = std::sin(config::PI / 2.0);
    assertNear(1.0, sin_half_pi, 1e-10, "sin(PI/2) should be 1");

    // Test default values are reasonable
    assertTrue(config::DEFAULT_BATCH_SIZE > 0,
               "DEFAULT_BATCH_SIZE should be positive");
    assertTrue(config::DEFAULT_BATCH_SIZE <= 128,
               "DEFAULT_BATCH_SIZE should be reasonable");

    assertTrue(config::DEFAULT_LEARNING_RATE > 0,
               "DEFAULT_LEARNING_RATE should be positive");
    assertTrue(config::DEFAULT_LEARNING_RATE < 1.0,
               "DEFAULT_LEARNING_RATE should be less than 1");

    assertTrue(config::DEFAULT_EPOCHS > 0, "DEFAULT_EPOCHS should be positive");
    assertTrue(config::DEFAULT_EPOCHS <= 10000,
               "DEFAULT_EPOCHS should be reasonable");
  }
};

/**
 * @class ConfigMathTest
 * @brief Test mathematical consistency of config values
 */
class ConfigMathTest : public TestCase {
public:
  ConfigMathTest() : TestCase("ConfigMathTest") {}

protected:
  void test() override {
    // Test that PI is consistent with mathematical properties
    double circumference = 2.0 * config::PI * 1.0;  // radius = 1
    double expected_circumference = 2.0 * config::PI;
    assertNear(expected_circumference, circumference, 1e-15,
               "Circumference calculation should be consistent");

    // Test that EPSILON is small enough for typical calculations
    assertTrue(config::EPSILON < 1e-6,
               "EPSILON should be smaller than typical precision requirements");
    assertTrue(config::EPSILON > 0.0, "EPSILON should be positive");

    // Test version number consistency
    int reconstructed_version = config::VERSION_MAJOR * 10000 +
                                config::VERSION_MINOR * 100 +
                                config::VERSION_PATCH;
    assertEqual(10000, reconstructed_version,
                "Version numbers should be consistent");

    // Test that default values are mathematically sensible
    assertTrue(config::DEFAULT_LEARNING_RATE * config::DEFAULT_EPOCHS < 100.0,
               "Learning rate and epochs combination should be reasonable");

    // Test that batch size is a power of 2 (common optimization)
    size_t batch_size = config::DEFAULT_BATCH_SIZE;
    bool is_power_of_2 = (batch_size & (batch_size - 1)) == 0;
    assertTrue(is_power_of_2,
               "DEFAULT_BATCH_SIZE should be a power of 2 for optimization");
  }
};

}  // namespace test
}  // namespace MLLib
