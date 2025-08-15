#include "../common/test_utils.hpp"
#include "../../include/MLLib.hpp"
#include <iostream>
#include <memory>

using namespace MLLib;

int main() {
    std::cout << "Running simple integration tests..." << std::endl;
    
    try {
        // Test 1: Basic model creation
        std::cout << "Test 1: Basic model creation..." << std::endl;
        
        auto model = std::make_unique<model::Sequential>();
        model->add(std::make_unique<layer::Dense>(3, 4));
        model->add(std::make_unique<layer::activation::ReLU>());
        model->add(std::make_unique<layer::Dense>(4, 1));
        model->add(std::make_unique<layer::activation::Sigmoid>());
        
        std::cout << "✓ Model created successfully" << std::endl;
        
        // Test 2: Basic prediction with vector
        std::cout << "Test 2: Prediction with vector..." << std::endl;
        
        std::vector<double> test_input = {0.2, 0.3, 0.4};
        auto prediction = model->predict(test_input);
        
        std::cout << "✓ Vector prediction completed successfully" << std::endl;
        std::cout << "Prediction size: " << prediction.size() << std::endl;
        
        // Test 3: Basic prediction with initializer list
        std::cout << "Test 3: Prediction with initializer list..." << std::endl;
        
        auto prediction2 = model->predict({0.2, 0.3, 0.4});
        
        std::cout << "✓ Initializer list prediction completed successfully" << std::endl;
        std::cout << "Prediction size: " << prediction2.size() << std::endl;
        
        std::cout << "\n🎉 Basic integration tests passed (including {} syntax)!" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Integration test failed: " << e.what() << std::endl;
        return 1;
    }
}
