/**
 * @file activation_gpu_example.cpp
 * @brief Example demonstrating the generic GPU activation system
 */

#include "../../include/MLLib/backend/gpu_kernel_manager.hpp"
#include "../../include/MLLib/backend/metal_backend.hpp"
#include <iostream>
#include <vector>
#include <cassert>

using namespace MLLib::Backend;

int main() {
    try {
        std::cout << "=== Generic GPU Activation System Demo ===" << std::endl;
        
        // Check if GPU is available
        if (!MetalBackend::isAvailable()) {
            std::cout << "Metal backend not available, exiting..." << std::endl;
            return 0;
        }
        
        // Initialize the system
        MetalBackend::initialize();
        std::cout << "✅ GPU system initialized successfully" << std::endl;
        
        // Test data
        const size_t size = 1000;
        std::vector<double> input(size);
        std::vector<double> output(size);
        
        // Initialize input with test values
        for (size_t i = 0; i < size; ++i) {
            input[i] = (static_cast<double>(i) / size) * 4.0 - 2.0; // Range: -2 to 2
        }
        
        std::cout << "\\nTesting built-in activation functions:" << std::endl;
        
        // Test ReLU
        std::cout << "Testing ReLU..." << std::endl;
        MetalBackend::relu(input.data(), output.data(), size);
        std::cout << "✅ ReLU completed. Sample: input=" << input[100] 
                  << " -> output=" << output[100] << std::endl;
        
        // Test Sigmoid
        std::cout << "Testing Sigmoid..." << std::endl;
        MetalBackend::sigmoid(input.data(), output.data(), size);
        std::cout << "✅ Sigmoid completed. Sample: input=" << input[500] 
                  << " -> output=" << output[500] << std::endl;
        
        // Test Tanh
        std::cout << "Testing Tanh..." << std::endl;
        MetalBackend::tanh_activation(input.data(), output.data(), size);
        std::cout << "✅ Tanh completed. Sample: input=" << input[750] 
                  << " -> output=" << output[750] << std::endl;
        
        // Test LeakyReLU with parameter
        std::cout << "Testing LeakyReLU (alpha=0.1)..." << std::endl;
        MetalBackend::leaky_relu(input.data(), output.data(), size, 0.1);
        std::cout << "✅ LeakyReLU completed. Sample: input=" << input[200] 
                  << " -> output=" << output[200] << std::endl;
        
        // Test GELU
        std::cout << "Testing GELU..." << std::endl;
        MetalBackend::gelu(input.data(), output.data(), size, false);
        std::cout << "✅ GELU completed. Sample: input=" << input[300] 
                  << " -> output=" << output[300] << std::endl;
        
        // Test ELU with parameter
        std::cout << "Testing ELU (alpha=1.0)..." << std::endl;
        MetalBackend::elu(input.data(), output.data(), size, 1.0);
        std::cout << "✅ ELU completed. Sample: input=" << input[400] 
                  << " -> output=" << output[400] << std::endl;
        
        // Test Swish
        std::cout << "Testing Swish..." << std::endl;
        MetalBackend::swish(input.data(), output.data(), size);
        std::cout << "✅ Swish completed. Sample: input=" << input[600] 
                  << " -> output=" << output[600] << std::endl;
        
        std::cout << "\\n=== Adding Custom Activation Function ===" << std::endl;
        
        // Register a new custom activation function
        ActivationKernelRegistry::registerActivation({
            "custom_sigmoid_scale",
            "1.0f / (1.0f + exp(-scale * input))",  // Scaled sigmoid
            {"scale"},
            true
        });
        
        std::cout << "✅ Custom activation 'custom_sigmoid_scale' registered" << std::endl;
        
        // Test the custom activation
        std::cout << "Testing custom scaled sigmoid (scale=2.0)..." << std::endl;
        ActivationKernelRegistry::executeActivation(
            "custom_sigmoid_scale", 
            input.data(), 
            output.data(), 
            size, 
            {2.0}  // scale parameter
        );
        std::cout << "✅ Custom activation completed. Sample: input=" << input[700] 
                  << " -> output=" << output[700] << std::endl;
        
        std::cout << "\\n=== Performance Comparison ===" << std::endl;
        
        // Simple performance test
        auto start = std::chrono::high_resolution_clock::now();
        
        constexpr int iterations = 100;
        for (int i = 0; i < iterations; ++i) {
            MetalBackend::relu(input.data(), output.data(), size);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        std::cout << "GPU ReLU (" << iterations << " iterations, " << size 
                  << " elements): " << duration.count() << " μs" << std::endl;
        std::cout << "Average per iteration: " << duration.count() / iterations << " μs" << std::endl;
        
        // Cleanup
        GPUKernelManager::cleanup();
        MetalBackend::cleanup();
        
        std::cout << "\\n🎉 All tests completed successfully!" << std::endl;
        std::cout << "\\nBenefits of the new system:" << std::endl;
        std::cout << "  ✅ Single implementation for all activation functions" << std::endl;
        std::cout << "  ✅ Easy to add new functions (just provide expression)" << std::endl;
        std::cout << "  ✅ Automatic parameter handling" << std::endl;
        std::cout << "  ✅ Reduced code duplication" << std::endl;
        std::cout << "  ✅ Consistent error handling" << std::endl;
        std::cout << "  ✅ Unified buffer management" << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return 1;
    }
}
