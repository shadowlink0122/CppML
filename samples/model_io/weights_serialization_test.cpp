/**
 * @file weights_serialization_test.cpp
 * @brief Test for improved weights and biases serialization
 */

#include <MLLib.hpp>
#include <iostream>
#include <memory>

using namespace MLLib;
using namespace MLLib::model;
using namespace MLLib::layer;

int main() {
    std::cout << "=== Model I/O Weights Serialization Test ===" << std::endl;
    
    // Create a simple neural network
    Sequential original_model;
    original_model.add(std::make_shared<Dense>(3, 4, true));  // input->hidden
    original_model.add(std::make_shared<Dense>(4, 2, true));  // hidden->output
    
    // Initialize with some test data
    NDArray test_input(std::vector<size_t>{1, 3});
    test_input[0] = 1.0;
    test_input[1] = 2.0;
    test_input[2] = 3.0;
    
    // Forward pass on original model
    NDArray original_output = original_model.predict(test_input);
    std::cout << "Original model output: [" << original_output[0] 
              << ", " << original_output[1] << "]" << std::endl;
    
    // Test 1: Generic Model I/O with weights serialization
    std::cout << "\n--- Test 1: Generic Model I/O ---" << std::endl;
    
    // Save using new generic interface with saved_models directory
    bool save_success = GenericModelIO::save_model(original_model, 
                                                   "saved_models/test_model_generic", 
                                                   SaveFormat::BINARY);
    std::cout << "Save (Generic Binary): " << (save_success ? "Success" : "Failed") << std::endl;
    
    // Load using new generic interface from saved_models directory
    auto loaded_model = GenericModelIO::load_model<Sequential>("saved_models/test_model_generic.bin", 
                                                               SaveFormat::BINARY);
    if (loaded_model) {
        NDArray loaded_output = loaded_model->predict(test_input);
        std::cout << "Loaded model output: [" << loaded_output[0] 
                  << ", " << loaded_output[1] << "]" << std::endl;
        
        // Compare outputs (should be identical)
        bool outputs_match = (std::abs(original_output[0] - loaded_output[0]) < 1e-10 &&
                             std::abs(original_output[1] - loaded_output[1]) < 1e-10);
        std::cout << "Weights preserved: " << (outputs_match ? "YES" : "NO") << std::endl;
    } else {
        std::cout << "Load (Generic Binary): Failed" << std::endl;
    }
    
    // Test 2: Legacy Model I/O comparison
    std::cout << "\n--- Test 2: Legacy Model I/O ---" << std::endl;
    
    bool legacy_save = ModelIO::save_model(original_model, "saved_models/test_model_legacy", SaveFormat::BINARY);
    std::cout << "Save (Legacy Binary): " << (legacy_save ? "Success" : "Failed") << std::endl;
    
    auto legacy_loaded = ModelIO::load_model("saved_models/test_model_legacy.bin", SaveFormat::BINARY);
    if (legacy_loaded) {
        NDArray legacy_output = legacy_loaded->predict(test_input);
        std::cout << "Legacy loaded output: [" << legacy_output[0] 
                  << ", " << legacy_output[1] << "]" << std::endl;
        
        bool legacy_match = (std::abs(original_output[0] - legacy_output[0]) < 1e-10 &&
                            std::abs(original_output[1] - legacy_output[1]) < 1e-10);
        std::cout << "Legacy weights preserved: " << (legacy_match ? "YES" : "NO") << std::endl;
    } else {
        std::cout << "Load (Legacy Binary): Failed" << std::endl;
    }
    
    // Test 3: Metadata and configuration
    std::cout << "\n--- Test 3: Model Metadata ---" << std::endl;
    
    auto metadata = original_model.get_serialization_metadata();
    std::cout << "Model type: " << static_cast<int>(metadata.model_type) << std::endl;
    std::cout << "Version: " << metadata.version << std::endl;
    std::cout << "Device: " << static_cast<int>(metadata.device) << std::endl;
    
    std::string config = original_model.get_config_string();
    std::cout << "Configuration: " << config << std::endl;
    
    std::cout << "\n=== Test Complete ===" << std::endl;
    
    return 0;
}
