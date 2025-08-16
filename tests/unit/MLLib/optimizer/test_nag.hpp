#pragma once

#include "../../../../include/MLLib/optimizer/nag.hpp"
#include "../../../common/test_utils.hpp"
#include <vector>

/**
 * @file test_nag.hpp
 * @brief Unit tests for NAG (Nesterov Accelerated Gradient) optimizer
 */

namespace MLLib {
namespace test {

class NAGTest : public TestCase {
public:
  NAGTest() : TestCase("NAG Test") {}
  
protected:
  void test() override {
    optimizer::NAG nag(0.01, 0.9);
    assertNear(nag.get_learning_rate(), 0.01, 1e-9, "NAG learning rate");
    
    NDArray param({2});
    param[0] = 1.0; param[1] = 2.0;
    NDArray grad({2});
    grad[0] = 0.1; grad[1] = 0.2;
    
    std::vector<NDArray*> parameters = {&param};
    std::vector<NDArray*> gradients = {&grad};
    
    double original_param0 = param[0];
    double original_param1 = param[1];
    
    nag.update(parameters, gradients);
    
    assertTrue(param[0] < original_param0, "Parameter should decrease");
    assertTrue(param[1] < original_param1, "Parameter should decrease");
  }
};

class NAGConstructorTest : public TestCase {
public:
  NAGConstructorTest() : TestCase("NAG Constructor Test") {}
  
protected:
  void test() override {
    // Test constructor with default momentum
    optimizer::NAG nag1(0.001);
    assertNear(nag1.get_learning_rate(), 0.001, 1e-9, "Default learning rate with default momentum");
    
    // Test custom parameters
    optimizer::NAG nag2(0.01, 0.9);
    assertNear(nag2.get_learning_rate(), 0.01, 1e-9, "Custom learning rate");
    
    // Test invalid parameters
    assertThrows<std::invalid_argument>([]() {
      optimizer::NAG nag(-0.1);
    }, "Should throw for negative learning rate");
    
    assertThrows<std::invalid_argument>([]() {
      optimizer::NAG nag(0.01, -0.1);
    }, "Should throw for negative momentum");
    
    assertThrows<std::invalid_argument>([]() {
      optimizer::NAG nag(0.01, 1.1);
    }, "Should throw for momentum > 1");
  }
};

class NAGMomentumTest : public TestCase {
public:
  NAGMomentumTest() : TestCase("NAG Momentum Test") {}
  
protected:
  void test() override {
    // Compare NAG with different momentum values
    optimizer::NAG nag_low_momentum(0.01, 0.1);
    optimizer::NAG nag_high_momentum(0.01, 0.9);
    
    // Create identical initial conditions
    NDArray param1({2});
    param1[0] = 1.0; param1[1] = 2.0;
    NDArray param2({2});
    param2[0] = 1.0; param2[1] = 2.0;
    
    NDArray grad({2});
    grad[0] = 0.1; grad[1] = 0.2;
    
    std::vector<NDArray*> parameters1 = {&param1};
    std::vector<NDArray*> parameters2 = {&param2};
    std::vector<NDArray*> gradients = {&grad};
    
    // First update
    nag_low_momentum.update(parameters1, gradients);
    nag_high_momentum.update(parameters2, gradients);
    
    // Second update (momentum should show effect here)
    nag_low_momentum.update(parameters1, gradients);
    nag_high_momentum.update(parameters2, gradients);
    
    // High momentum should result in larger parameter changes
    assertTrue(std::abs(param2[0] - 1.0) > std::abs(param1[0] - 1.0), 
               "High momentum should cause larger parameter changes");
    assertTrue(std::abs(param2[1] - 2.0) > std::abs(param1[1] - 2.0), 
               "High momentum should cause larger parameter changes");
  }
};

class NAGResetTest : public TestCase {
public:
  NAGResetTest() : TestCase("NAG Reset Test") {}
  
protected:
  void test() override {
    optimizer::NAG nag(0.01, 0.9);
    
    NDArray param({2});
    param[0] = 1.0; param[1] = 2.0;
    NDArray grad({2});
    grad[0] = 0.1; grad[1] = 0.2;
    
    std::vector<NDArray*> parameters = {&param};
    std::vector<NDArray*> gradients = {&grad};
    
    // Update once to build up momentum
    nag.update(parameters, gradients);
    
    // Reset optimizer
    nag.reset();
    
    assertTrue(true, "NAG reset completed without error");
  }
};

}  // namespace test
}  // namespace MLLib
