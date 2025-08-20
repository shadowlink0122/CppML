#pragma once

#include "../../../../include/MLLib/layer/activation/relu.hpp"
#include "../../../../include/MLLib/layer/activation/sigmoid.hpp"
#include "../../../../include/MLLib/layer/activation/tanh.hpp"
#include "../../../../include/MLLib/layer/dense.hpp"
#include "../../../../include/MLLib/model/model_io.hpp"
#include "../../../../include/MLLib/model/sequential.hpp"
#include "../../../../include/MLLib/ndarray.hpp"
#include "../../../common/test_utils.hpp"
#include <chrono>
#include <filesystem>
#include <random>
#include <thread>

namespace MLLib {
namespace test {

/**
 * @class SequentialModelProductionWorkflowTest
 * @brief Test Sequential model in production-like workflow scenarios
 */
class SequentialModelProductionWorkflowTest : public TestCase {
public:
  SequentialModelProductionWorkflowTest()
      : TestCase("SequentialModelProductionWorkflowTest") {}

protected:
  void test() override {
    using namespace MLLib::model;
    using namespace MLLib::layer;

    printf("Testing Sequential model production workflow...\n");

    std::string temp_dir = createTempDirectory();
    const int num_samples = 50;
    int successful_operations = 0;
    std::vector<std::pair<std::string, double>> model_performances;

    try {
      // Create various Sequential models with different architectures
      std::vector<std::vector<int>> architectures = {
          {2, 4, 3, 1},       // Simple classification
          {3, 8, 5, 2},       // Medium network
          {4, 10, 8, 6, 3},   // Deep network
          {5, 15, 10, 5, 1},  // Wide to narrow
          {1, 3, 5, 3, 1}     // Autoencoder-like
      };

      for (size_t arch_idx = 0; arch_idx < architectures.size(); ++arch_idx) {
        const auto& arch = architectures[arch_idx];
        printf("  Testing architecture %zu: ", arch_idx + 1);
        for (size_t i = 0; i < arch.size(); ++i) {
          printf("%d", arch[i]);
          if (i < arch.size() - 1) printf("-");
        }
        printf("\n");

        for (size_t sample = 0; sample < num_samples / architectures.size();
             ++sample) {
          // Create Sequential model
          auto model = std::make_unique<Sequential>();

          // Add layers
          for (size_t i = 0; i < arch.size() - 1; ++i) {
            model->add(std::make_unique<Dense>(arch[i], arch[i + 1]));
            if (i < arch.size() - 2) {  // Don't add activation after last layer
              switch (i % 3) {
              case 0: model->add(std::make_unique<activation::ReLU>()); break;
              case 1:
                model->add(std::make_unique<activation::Sigmoid>());
                break;
              case 2: model->add(std::make_unique<activation::Tanh>()); break;
              }
            }
          }

          // Create test data
          NDArray input({1, static_cast<size_t>(arch[0])});
          std::random_device rd;
          std::mt19937 gen(42 + sample + arch_idx * 1000);
          std::uniform_real_distribution<double> dis(-1.0, 1.0);

          for (int i = 0; i < arch[0]; ++i) {
            input[i] = dis(gen);
          }

          // Test forward pass
          auto start_time = std::chrono::high_resolution_clock::now();
          NDArray original_output = model->predict(input);
          auto end_time = std::chrono::high_resolution_clock::now();
          auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
              end_time - start_time);
          double forward_time_ms = duration.count() / 1000.0;

          // Save model
          std::string model_path = temp_dir + "/prod_sequential_" +
              std::to_string(arch_idx) + "_" + std::to_string(sample);

          start_time = std::chrono::high_resolution_clock::now();
          bool save_success = GenericModelIO::save_model(*model, model_path,
                                                         SaveFormat::BINARY);
          end_time = std::chrono::high_resolution_clock::now();
          duration = std::chrono::duration_cast<std::chrono::microseconds>(
              end_time - start_time);
          double save_time_ms = duration.count() / 1000.0;

          if (!save_success) {
            printf("    ❌ Save failed for sample %zu\n", sample);
            continue;
          }

          // Load model
          start_time = std::chrono::high_resolution_clock::now();
          auto loaded_model =
              GenericModelIO::load_model<Sequential>(model_path,
                                                     SaveFormat::BINARY);
          end_time = std::chrono::high_resolution_clock::now();
          duration = std::chrono::duration_cast<std::chrono::microseconds>(
              end_time - start_time);
          double load_time_ms = duration.count() / 1000.0;

          if (loaded_model == nullptr) {
            printf("    ❌ Load failed for sample %zu\n", sample);
            continue;
          }

          // Test loaded model
          start_time = std::chrono::high_resolution_clock::now();
          NDArray loaded_output = loaded_model->predict(input);
          end_time = std::chrono::high_resolution_clock::now();
          duration = std::chrono::duration_cast<std::chrono::microseconds>(
              end_time - start_time);
          double loaded_forward_time_ms = duration.count() / 1000.0;

          // Verify outputs match
          bool outputs_match = true;
          double tolerance = 1e-10;
          for (size_t i = 0; i < original_output.size(); ++i) {
            if (std::abs(original_output[i] - loaded_output[i]) > tolerance) {
              outputs_match = false;
              break;
            }
          }

          if (outputs_match) {
            successful_operations++;
            double total_time = forward_time_ms + save_time_ms + load_time_ms +
                loaded_forward_time_ms;
            model_performances.push_back({"arch_" + std::to_string(arch_idx) +
                                              "_sample_" +
                                              std::to_string(sample),
                                          total_time});
          } else {
            printf("    ❌ Output mismatch for sample %zu\n", sample);
          }
        }
      }

      // Calculate statistics
      double success_rate = (double)successful_operations / num_samples * 100.0;
      assertTrue(success_rate >= 95.0, "Success rate should be >= 95%");

      if (!model_performances.empty()) {
        double total_time = 0.0;
        double min_time = model_performances[0].second;
        double max_time = model_performances[0].second;

        for (const auto& perf : model_performances) {
          total_time += perf.second;
          min_time = std::min(min_time, perf.second);
          max_time = std::max(max_time, perf.second);
        }

        double avg_time = total_time / model_performances.size();

        printf("  Production workflow statistics:\n");
        printf("    Success rate: %.1f%% (%d/%d)\n", success_rate,
               successful_operations, num_samples);
        printf("    Average processing time: %.2fms\n", avg_time);
        printf("    Min processing time: %.2fms\n", min_time);
        printf("    Max processing time: %.2fms\n", max_time);

        assertTrue(avg_time < 10.0, "Average processing time should be < 10ms");
      }

    } catch (const std::exception& e) {
      printf("  ❌ Production workflow test failed: %s\n", e.what());
      assertTrue(false, "Production workflow should not throw exceptions");
    }

    removeTempDirectory(temp_dir);
    printf(
        "  Sequential model production workflow test completed successfully\n");
  }
};

/**
 * @class SequentialModelRobustnessTest
 * @brief Test Sequential model robustness with edge cases
 */
class SequentialModelRobustnessTest : public TestCase {
public:
  SequentialModelRobustnessTest() : TestCase("SequentialModelRobustnessTest") {}

protected:
  void test() override {
    using namespace MLLib::model;
    using namespace MLLib::layer;

    printf("Testing Sequential model robustness...\n");

    std::string temp_dir = createTempDirectory();

    // Test 1: Minimal model (single layer)
    {
      printf("  Testing minimal Sequential model...\n");
      auto minimal_model = std::make_unique<Sequential>();
      minimal_model->add(std::make_unique<Dense>(1, 1));

      NDArray minimal_input({1, 1});
      minimal_input[0] = 42.0;
      NDArray minimal_output = minimal_model->predict(minimal_input);

      std::string minimal_path = temp_dir + "/minimal_sequential";
      assertTrue(GenericModelIO::save_model(*minimal_model, minimal_path,
                                            SaveFormat::BINARY),
                 "Minimal model should save");

      auto loaded_minimal =
          GenericModelIO::load_model<Sequential>(minimal_path,
                                                 SaveFormat::BINARY);
      assertTrue(loaded_minimal != nullptr, "Minimal model should load");

      NDArray loaded_minimal_output = loaded_minimal->predict(minimal_input);
      assertTrue(std::abs(minimal_output[0] - loaded_minimal_output[0]) < 1e-10,
                 "Minimal model outputs should match");
    }

    // Test 2: Large input dimensions
    {
      printf("  Testing Sequential model with large dimensions...\n");
      auto large_model = std::make_unique<Sequential>();
      large_model->add(std::make_unique<Dense>(100, 50));
      large_model->add(std::make_unique<activation::ReLU>());
      large_model->add(std::make_unique<Dense>(50, 10));

      NDArray large_input({1, 100});
      std::random_device rd;
      std::mt19937 gen(123);
      std::uniform_real_distribution<double> dis(-5.0, 5.0);

      for (int i = 0; i < 100; ++i) {
        large_input[i] = dis(gen);
      }

      NDArray large_output = large_model->predict(large_input);

      std::string large_path = temp_dir + "/large_sequential";
      assertTrue(GenericModelIO::save_model(*large_model, large_path,
                                            SaveFormat::BINARY),
                 "Large model should save");

      // Verify file size is reasonable
      size_t file_size = std::filesystem::file_size(large_path + ".bin");
      assertTrue(file_size > 1000, "Large model file should be substantial");

      auto loaded_large =
          GenericModelIO::load_model<Sequential>(large_path,
                                                 SaveFormat::BINARY);
      assertTrue(loaded_large != nullptr, "Large model should load");

      NDArray loaded_large_output = loaded_large->predict(large_input);

      bool large_match = true;
      for (size_t i = 0; i < large_output.size(); ++i) {
        if (std::abs(large_output[i] - loaded_large_output[i]) > 1e-10) {
          large_match = false;
          break;
        }
      }
      assertTrue(large_match, "Large model outputs should match");
    }

    // Test 3: Edge case values
    {
      printf("  Testing Sequential model with edge case values...\n");
      auto edge_model = std::make_unique<Sequential>();
      edge_model->add(std::make_unique<Dense>(3, 2));
      edge_model->add(std::make_unique<activation::Sigmoid>());

      // Test with extreme values
      NDArray extreme_input({1, 3});
      extreme_input[0] = 1000.0;
      extreme_input[1] = -1000.0;
      extreme_input[2] = 0.0;

      NDArray extreme_output = edge_model->predict(extreme_input);

      std::string edge_path = temp_dir + "/edge_sequential";
      assertTrue(GenericModelIO::save_model(*edge_model, edge_path,
                                            SaveFormat::BINARY),
                 "Edge case model should save");

      auto loaded_edge =
          GenericModelIO::load_model<Sequential>(edge_path, SaveFormat::BINARY);
      assertTrue(loaded_edge != nullptr, "Edge case model should load");

      NDArray loaded_extreme_output = loaded_edge->predict(extreme_input);

      bool extreme_match = true;
      for (size_t i = 0; i < extreme_output.size(); ++i) {
        if (std::abs(extreme_output[i] - loaded_extreme_output[i]) > 1e-10) {
          extreme_match = false;
          break;
        }
      }
      assertTrue(extreme_match, "Edge case model outputs should match");
    }

    removeTempDirectory(temp_dir);
    printf("  Sequential model robustness test completed successfully\n");
  }
};

/**
 * @class SequentialModelConcurrentAccessTest
 * @brief Test concurrent access to Sequential model I/O
 */
class SequentialModelConcurrentAccessTest : public TestCase {
public:
  SequentialModelConcurrentAccessTest()
      : TestCase("SequentialModelConcurrentAccessTest") {}

protected:
  void test() override {
    using namespace MLLib::model;
    using namespace MLLib::layer;

    printf("Testing Sequential model concurrent access...\n");

    std::string temp_dir = createTempDirectory();
    const int num_threads = 4;
    const int operations_per_thread = 5;
    std::vector<std::thread> threads;
    std::vector<bool> thread_results(num_threads, false);

    // Create base model
    auto base_model = std::make_unique<Sequential>();
    base_model->add(std::make_unique<Dense>(3, 4));
    base_model->add(std::make_unique<activation::ReLU>());
    base_model->add(std::make_unique<Dense>(4, 2));

    // Save base model
    std::string base_path = temp_dir + "/base_concurrent_sequential";
    assertTrue(GenericModelIO::save_model(*base_model, base_path,
                                          SaveFormat::BINARY),
               "Base concurrent model should save");

    // Launch concurrent operations
    for (int t = 0; t < num_threads; ++t) {
      threads.emplace_back([&, t]() {
        try {
          bool all_ops_successful = true;

          for (int op = 0; op < operations_per_thread; ++op) {
            // Load model
            auto thread_model =
                GenericModelIO::load_model<Sequential>(base_path,
                                                       SaveFormat::BINARY);
            if (thread_model == nullptr) {
              all_ops_successful = false;
              break;
            }

            // Test model
            NDArray thread_input({1, 3});
            thread_input[0] = static_cast<double>(t);
            thread_input[1] = static_cast<double>(op);
            thread_input[2] = static_cast<double>(t * op);

            NDArray thread_output = thread_model->predict(thread_input);
            if (thread_output.size() != 2) {
              all_ops_successful = false;
              break;
            }

            // Save thread-specific model
            std::string thread_path = temp_dir + "/thread_" +
                std::to_string(t) + "_" + std::to_string(op);
            if (!GenericModelIO::save_model(*thread_model, thread_path,
                                            SaveFormat::BINARY)) {
              all_ops_successful = false;
              break;
            }

            // Small delay to increase chance of concurrent access
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
          }

          thread_results[t] = all_ops_successful;
        } catch (...) {
          thread_results[t] = false;
        }
      });
    }

    // Wait for all threads to complete
    for (auto& thread : threads) {
      thread.join();
    }

    // Verify all threads succeeded
    bool all_threads_succeeded = true;
    for (int t = 0; t < num_threads; ++t) {
      if (!thread_results[t]) {
        printf("  ❌ Thread %d failed\n", t);
        all_threads_succeeded = false;
      }
    }

    assertTrue(all_threads_succeeded,
               "All concurrent operations should succeed");

    // Verify all thread-specific files were created
    int created_files = 0;
    for (int t = 0; t < num_threads; ++t) {
      for (int op = 0; op < operations_per_thread; ++op) {
        std::string thread_file = temp_dir + "/thread_" + std::to_string(t) +
            "_" + std::to_string(op) + ".bin";
        if (std::filesystem::exists(thread_file)) {
          created_files++;
        }
      }
    }

    int expected_files = num_threads * operations_per_thread;
    assertTrue(created_files == expected_files,
               "All concurrent files should be created (" +
                   std::to_string(created_files) + "/" +
                   std::to_string(expected_files) + ")");

    printf(
        "  Concurrent access test: %d threads × %d ops = %d total operations completed successfully\n",
        num_threads, operations_per_thread, expected_files);

    removeTempDirectory(temp_dir);
    printf(
        "  Sequential model concurrent access test completed successfully\n");
  }
};

}  // namespace test
}  // namespace MLLib
