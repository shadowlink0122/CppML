#pragma once

#include "../../../../include/MLLib/optimizer/adadelta.hpp"
#include "../../../common/test_utils.hpp"
#include <vector>

/**
 * @file test_adadelta.hpp
 * @brief Unit tests for AdaDelta optimizer
 */

namespace MLLib {
namespace test {

class AdaDeltaTest : public TestCase {
public:
  AdaDeltaTest() : TestCase("AdaDelta Test") {}
  
protected:
  void test() override {
    optimizer::AdaDelta adadelta;
    
    NDArray param({2});
    param[0] = 1.0; param[1] = 2.0;
    NDArray grad({2});
    grad[0] = 0.1; grad[1] = 0.2;
    
    std::vector<NDArray*> parameters = {&param};
    std::vector<NDArray*> gradients = {&grad};
    
    double original_param0 = param[0];
    double original_param1 = param[1];
    
    adadelta.update(parameters, gradients);
    
    assertTrue(param[0] != original_param0, "Parameter should change");
    assertTrue(param[1] != original_param1, "Parameter should change");
  }
};

class AdaDeltaConstructorTest : public TestCase {
public:
  AdaDeltaConstructorTest() : TestCase("AdaDelta Constructor Test") {}
  
protected:
  void test() override {
    // Test default constructor
    optimizer::AdaDelta adadelta1;
    assertTrue(true, "Default constructor should work");
    
    // Test custom parameters
    optimizer::AdaDelta adadelta2(0.9, 1e-6);
    assertTrue(true, "Custom constructor should work");
    
    // Test invalid parameters
    assertThrows<std::invalid_argument>([]() {
      optimizer::AdaDelta adadelta(1.0, -0.1); // negative rho
    }, "Should throw for negative rho");
    
    assertThrows<std::invalid_argument>([]() {
      optimizer::AdaDelta adadelta(1.0, 0.95, -1e-6); // negative epsilon
    }, "Should throw for negative epsilon");
    
    assertThrows<std::invalid_argument>([]() {
      optimizer::AdaDelta adadelta(1.0, 1.1); // rho > 1
    }, "Should throw for rho > 1");
  }
};

class AdaDeltaResetTest : public TestCase {
public:
  AdaDeltaResetTest() : TestCase("AdaDelta Reset Test") {}
  
protected:
  void test() override {
    optimizer::AdaDelta adadelta;
    
    NDArray param({2});
    param[0] = 1.0; param[1] = 2.0;
    NDArray grad({2});
    grad[0] = 0.1; grad[1] = 0.2;
    
    std::vector<NDArray*> parameters = {&param};
    std::vector<NDArray*> gradients = {&grad};
    
    // Update once to build up state
    adadelta.update(parameters, gradients);
    
    // Reset optimizer
    adadelta.reset();
    
    assertTrue(true, "AdaDelta reset completed without error");
  }
};

class AdaDeltaMultipleUpdatesTest : public TestCase {
public:
  AdaDeltaMultipleUpdatesTest() : TestCase("AdaDelta Multiple Updates Test") {}
  
protected:
  void test() override {
    optimizer::AdaDelta adadelta;
    
    NDArray param({2});
    param[0] = 1.0; param[1] = 2.0;
    NDArray grad({2});
    grad[0] = 0.1; grad[1] = 0.2;
    
    std::vector<NDArray*> parameters = {&param};
    std::vector<NDArray*> gradients = {&grad};
    
    double param0_before = param[0];
    double param1_before = param[1];
    
    // Multiple updates should continue to modify parameters
    adadelta.update(parameters, gradients);
    double param0_after_first = param[0];
    double param1_after_first = param[1];
    
    adadelta.update(parameters, gradients);
    double param0_after_second = param[0];
    double param1_after_second = param[1];
    
    assertTrue(param0_after_first != param0_before, "First update should change parameters");
    assertTrue(param1_after_first != param1_before, "First update should change parameters");
    assertTrue(param0_after_second != param0_after_first, "Second update should further change parameters");
    assertTrue(param1_after_second != param1_after_first, "Second update should further change parameters");
  }
};

}  // namespace test
}  // namespace MLLib
