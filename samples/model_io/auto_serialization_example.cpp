/**
 * @file auto_serialization_example.cpp
 * @brief Example demonstrating the model I/O serialization system
 * 
 * This example shows how to use the binary model I/O system for
 * persistent model storage with Sequential models.
 */

#include <MLLib.hpp>
#include <iostream>
#include <memory>

using namespace MLLib;
using namespace MLLib::model;
using namespace MLLib::layer;

/**
 * @brief Create a simple neural network for testing serialization
 */
std::unique_ptr<Sequential> create_test_model(int input_size, int hidden_size, int output_size) {
    auto model = std::make_unique<Sequential>();
    
    // Add layers
    auto hidden = std::make_shared<Dense>(input_size, hidden_size);
    auto output = std::make_shared<Dense>(hidden_size, output_size);
    
    model->add(hidden);
    model->add(output);
    
    return model;
}

/**
 * @brief Demonstrate Sequential model serialization
 */
void test_sequential_model_io() {
    std::cout << "=== Testing Sequential Model I/O ===\n";
    
    // Create and initialize model
    auto original_model = create_test_model(3, 5, 2);
    
    // Initialize with some test data
    NDArray input(std::vector<size_t>{1, 3});
    input[0] = 1.0;
    input[1] = 2.0;
    input[2] = 3.0;
    
    auto original_output = original_model->predict(input);
    
    std::cout << "Original model output: [";
    std::cout << original_output[0] << ", " << original_output[1];
    std::cout << "]" << std::endl;
    
    // Save model using generic I/O to saved_models directory
    std::string model_path = "saved_models/test_sequential_model";
    bool save_success = GenericModelIO::save_model(*original_model, model_path, SaveFormat::BINARY);
    std::cout << "Save success: " << (save_success ? "✅" : "❌") << std::endl;
    
    if (!save_success) {
        std::cout << "Failed to save model" << std::endl;
        return;
    }
    
    // Load model using generic I/O from saved_models directory
    auto loaded_model = GenericModelIO::load_model<Sequential>(model_path + ".bin", SaveFormat::BINARY);
    std::cout << "Load success: " << (loaded_model ? "✅" : "❌") << std::endl;
    
    if (!loaded_model) {
        std::cout << "Failed to load model" << std::endl;
        return;
    }
    
    // Test loaded model
    auto loaded_output = loaded_model->predict(input);
    std::cout << "Loaded model output: [";
    std::cout << loaded_output[0] << ", " << loaded_output[1];
    std::cout << "]" << std::endl;
    
    // Compare outputs
    double max_diff = std::max(
        std::abs(original_output[0] - loaded_output[0]),
        std::abs(original_output[1] - loaded_output[1])
    );
    
    std::cout << "Maximum difference: " << max_diff << std::endl;
    if (max_diff < 1e-10) {
        std::cout << "✅ Model I/O test passed!" << std::endl;
    } else {
        std::cout << "❌ Model I/O test failed!" << std::endl;
    }
}

int main() {
    std::cout << "=== Model I/O System Demo ===" << std::endl;
    
    try {
        // Test Sequential model I/O
        test_sequential_model_io();
        
        std::cout << "\n=== Demo completed successfully! ===\n" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "❌ Demo failed: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
