#include "../../include/MLLib.hpp"
#include "../common/test_utils.hpp"
#include <cstdio>
#include <memory>

using namespace MLLib;

int main() {
  printf("Running simple integration tests...\n");

  try {
    // Test 1: Basic model creation
    printf("Test 1: Basic model creation...\n");

    auto model = std::make_unique<model::Sequential>();
    model->add(std::make_unique<layer::Dense>(3, 4));
    model->add(std::make_unique<layer::activation::ReLU>());
    model->add(std::make_unique<layer::Dense>(4, 1));
    model->add(std::make_unique<layer::activation::Sigmoid>());

    printf("✅ Model created successfully\n");

    // Test 2: Basic prediction with vector
    printf("Test 2: Prediction with vector...\n");

    std::vector<double> test_input = {0.2, 0.3, 0.4};
    auto prediction = model->predict(test_input);

    printf("✅ Vector prediction completed successfully\n");
    printf("Prediction size: %zu\n", prediction.size());

    // Test 3: Basic prediction with initializer list
    printf("Test 3: Prediction with initializer list...\n");

    auto prediction2 = model->predict({0.2, 0.3, 0.4});

    printf("✅ Initializer list prediction completed successfully\n");
    printf("Prediction size: %zu\n", prediction2.size());

    printf("\n🎉 Basic integration tests passed (including {} syntax)!\n");
    return 0;

  } catch (const std::exception& e) {
    printf("❌ Integration test failed: %s\n", e.what());
    return 1;
  }
}
