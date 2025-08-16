#include "../common/test_utils.hpp"
#include "MLLib/backend/test_gpu_backend.hpp"
#include "MLLib/layer/activation/test_activation.hpp"
#include "MLLib/layer/activation/test_elu.hpp"
#include "MLLib/layer/activation/test_gelu.hpp"
#include "MLLib/layer/activation/test_leaky_relu.hpp"
#include "MLLib/layer/activation/test_softmax.hpp"
#include "MLLib/layer/activation/test_swish.hpp"
#include "MLLib/layer/test_dense.hpp"
#include "MLLib/model/test_model_io.hpp"
#include "MLLib/model/test_sequential.hpp"
#include "MLLib/optimizer/test_adadelta.hpp"
#include "MLLib/optimizer/test_adagrad.hpp"
#include "MLLib/optimizer/test_adam.hpp"
#include "MLLib/optimizer/test_nag.hpp"
#include "MLLib/optimizer/test_rmsprop.hpp"
#include "MLLib/test_config.hpp"
#include "MLLib/test_ndarray.hpp"
#include <chrono>
#include <iomanip>

/**
 * @file unit_test_main.cpp
 * @brief Main entry point for unit tests
 *
 * This file contains all unit tests for the MLLib library.
 * Tests are organized by component and run in a structured manner.
 */

int main() {
  using namespace MLLib::test;

  std::cout << "=== MLLib Unit Test Suite ===" << std::endl;
  std::cout << "Running comprehensive unit tests for MLLib v1.0.0" << std::endl;
  std::cout << "Test execution with output capture enabled" << std::endl;
  std::cout << std::endl;

  // Start timing the entire test suite
  auto suite_start_time = std::chrono::high_resolution_clock::now();

  bool all_tests_passed = true;
  int total_tests = 0;
  int passed_tests = 0;
  double total_execution_time = 0.0;

  // Helper function to run test and update counters
  auto runTest = [&](std::unique_ptr<TestCase> test) {
    total_tests++;
    bool result = test->run();
    if (result) passed_tests++;
    all_tests_passed &= result;
    total_execution_time += test->getExecutionTimeMs();
  };

  // Config tests
  std::cout << "\n--- Config Module Tests ---" << std::endl;
  runTest(std::make_unique<ConfigConstantsTest>());
  runTest(std::make_unique<ConfigUsageTest>());
  runTest(std::make_unique<ConfigMathTest>());

  // NDArray tests
  std::cout << "\n--- NDArray Module Tests ---" << std::endl;
  runTest(std::make_unique<NDArrayConstructorTest>());
  runTest(std::make_unique<NDArrayAccessTest>());
  runTest(std::make_unique<NDArrayOperationsTest>());
  runTest(std::make_unique<NDArrayArithmeticTest>());
  runTest(std::make_unique<NDArrayMatmulTest>());
  runTest(std::make_unique<NDArrayErrorTest>());

  // Dense layer tests
  std::cout << "\n--- Dense Layer Tests ---" << std::endl;
  runTest(std::make_unique<DenseConstructorTest>());
  runTest(std::make_unique<DenseForwardTest>());
  runTest(std::make_unique<DenseBackwardTest>());
  runTest(std::make_unique<DenseParameterTest>());

  // Activation function tests
  std::cout << "\n--- Activation Function Tests ---" << std::endl;
  runTest(std::make_unique<ReLUTest>());
  runTest(std::make_unique<ReLUBackwardTest>());
  runTest(std::make_unique<SigmoidTest>());
  runTest(std::make_unique<SigmoidBackwardTest>());
  runTest(std::make_unique<TanhTest>());
  runTest(std::make_unique<TanhBackwardTest>());
  runTest(std::make_unique<ActivationErrorTest>());

  // New activation function tests
  std::cout << "\n--- New Activation Function Tests ---" << std::endl;
  runTest(std::make_unique<LeakyReLUTest>());
  runTest(std::make_unique<LeakyReLUErrorTest>());
  runTest(std::make_unique<ELUTest>());
  runTest(std::make_unique<ELUErrorTest>());
  runTest(std::make_unique<SwishTest>());
  runTest(std::make_unique<SwishErrorTest>());
  runTest(std::make_unique<GELUTest>());
  runTest(std::make_unique<GELUApproximateTest>());
  runTest(std::make_unique<GELUErrorTest>());
  runTest(std::make_unique<SoftmaxTest>());
  runTest(std::make_unique<SoftmaxBatchTest>());
  runTest(std::make_unique<SoftmaxErrorTest>());

  // Optimizer tests
  std::cout << "\n--- Optimizer Tests ---" << std::endl;
  runTest(std::make_unique<AdamConstructorTest>());
  runTest(std::make_unique<AdamUpdateTest>());
  runTest(std::make_unique<AdamResetTest>());
  runTest(std::make_unique<AdamErrorTest>());
  runTest(std::make_unique<RMSpropConstructorTest>());
  runTest(std::make_unique<RMSpropUpdateTest>());
  runTest(std::make_unique<RMSpropResetTest>());
  runTest(std::make_unique<AdaGradTest>());
  runTest(std::make_unique<AdaGradConstructorTest>());
  runTest(std::make_unique<AdaGradResetTest>());
  runTest(std::make_unique<AdaDeltaTest>());
  runTest(std::make_unique<AdaDeltaConstructorTest>());
  runTest(std::make_unique<AdaDeltaResetTest>());
  runTest(std::make_unique<AdaDeltaMultipleUpdatesTest>());
  runTest(std::make_unique<NAGTest>());
  runTest(std::make_unique<NAGConstructorTest>());
  runTest(std::make_unique<NAGMomentumTest>());
  runTest(std::make_unique<NAGResetTest>());

  // Sequential model tests
  std::cout << "\n--- Sequential Model Tests ---" << std::endl;
  runTest(std::make_unique<SequentialModelTests>());

  // GPU backend tests
  std::cout << "\n--- GPU Backend Tests ---" << std::endl;
  runTest(std::make_unique<GPUAvailabilityTest>());
  runTest(std::make_unique<GPUDeviceValidationTest>());
  runTest(std::make_unique<GPUBackendOperationsTest>());
  runTest(std::make_unique<GPUArrayOperationsTest>());
  runTest(std::make_unique<GPUModelTest>());
  runTest(std::make_unique<GPUPerformanceTest>());

  // Model I/O tests
  std::cout << "\n--- Model I/O Tests ---" << std::endl;
  runTest(std::make_unique<ModelFormatTest>());
  runTest(std::make_unique<ModelSaveLoadTest>());
  runTest(std::make_unique<ModelParameterTest>());
  runTest(std::make_unique<ModelIOErrorTest>());
  runTest(std::make_unique<ModelIOFileHandlingTest>());

  // Print final summary
  std::cout << "\n" << std::string(60, '=') << std::endl;
  std::cout << "FINAL TEST SUMMARY" << std::endl;
  std::cout << std::string(60, '=') << std::endl;
  std::cout << "Total individual tests: " << total_tests << std::endl;
  std::cout << "Passed tests: " << passed_tests << std::endl;
  std::cout << "Failed tests: " << (total_tests - passed_tests) << std::endl;

  // Calculate total suite execution time
  auto suite_end_time = std::chrono::high_resolution_clock::now();
  auto suite_duration = std::chrono::duration_cast<std::chrono::microseconds>(
      suite_end_time - suite_start_time);
  double suite_time_ms = suite_duration.count() / 1000.0;

  std::cout << std::fixed << std::setprecision(2);
  std::cout << "Total test execution time: " << total_execution_time << "ms"
            << std::endl;
  std::cout << "Total suite time (including overhead): " << suite_time_ms
            << "ms" << std::endl;
  std::cout << std::endl;
  if (all_tests_passed) {
    std::cout << "🎉 ALL UNIT TESTS PASSED! 🎉" << std::endl;
    std::cout << "MLLib is ready for production use." << std::endl;
  } else {
    std::cout << "❌ SOME UNIT TESTS FAILED" << std::endl;
    std::cout << "Please review the test output above and fix the issues."
              << std::endl;
  }

  std::cout << std::string(60, '=') << std::endl;

  return all_tests_passed ? 0 : 1;
}
