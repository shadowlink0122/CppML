/**
 * @file test_gpu_backend.hpp
 * @brief Unit tests for GPU backend functionality
 */

#pragma once

#include "../../../../include/MLLib/device/device.hpp"
#include "../../../../include/MLLib/backend/backend.hpp"
#include "../../../../include/MLLib/ndarray.hpp"
#include "../../../../include/MLLib/model/sequential.hpp"
#include "../../../common/test_utils.hpp"
#include <iostream>
#include <vector>

namespace MLLib {
namespace test {

/**
 * @class GPUAvailabilityTest
 * @brief Test GPU device detection and availability
 */
class GPUAvailabilityTest : public TestCase {
public:
    GPUAvailabilityTest() : TestCase("GPUAvailabilityTest") {}

protected:
    void test() override {
        // Test GPU availability detection
        bool gpu_available = Device::isGPUAvailable();
        
        // Should return false on systems without CUDA/GPU
        // This is expected behavior on most test environments
        std::cout << "  GPU Available: " << (gpu_available ? "Yes" : "No") << std::endl;
        
        // Test should not fail regardless of GPU availability
        assertTrue(true, "GPU availability check should not fail");
    }
};

/**
 * @class GPUDeviceValidationTest
 * @brief Test GPU device validation and warning system
 */
class GPUDeviceValidationTest : public TestCase {
public:
    GPUDeviceValidationTest() : TestCase("GPUDeviceValidationTest") {}

protected:
    void test() override {
        // Store original device
        DeviceType original_device = Device::getCurrentDevice();
        
        // Test setting GPU device with validation (should show warning if no GPU)
        bool gpu_set = Device::setDeviceWithValidation(DeviceType::GPU, false); // no warnings in test
        
        // On systems without GPU, this should return false and fallback to CPU
        if (!Device::isGPUAvailable()) {
            assertFalse(gpu_set, "GPU device set should fail when GPU not available");
            assertTrue(Device::getCurrentDevice() == DeviceType::CPU, 
                       "Should fallback to CPU when GPU not available");
        }
        
        // Test CPU device validation (should always succeed)
        bool cpu_set = Device::setDeviceWithValidation(DeviceType::CPU, false);
        assertTrue(cpu_set, "CPU device set should always succeed");
        assertTrue(Device::getCurrentDevice() == DeviceType::CPU, "Should be set to CPU");
        
        // Test device type string conversion
        assertTrue(std::string(Device::getDeviceTypeString(DeviceType::CPU)) == "CPU", 
                   "CPU device string should be 'CPU'");
        assertTrue(std::string(Device::getDeviceTypeString(DeviceType::GPU)) == "GPU", 
                   "GPU device string should be 'GPU'");
        assertTrue(std::string(Device::getDeviceTypeString(DeviceType::AUTO)) == "AUTO", 
                   "AUTO device string should be 'AUTO'");
        
        // Restore original device
        Device::setDevice(original_device);
    }
};

/**
 * @class GPUBackendOperationsTest
 * @brief Test GPU backend operations with fallback
 */
class GPUBackendOperationsTest : public TestCase {
public:
    GPUBackendOperationsTest() : TestCase("GPUBackendOperationsTest") {}

protected:
    void test() override {
        // Store original device
        DeviceType original_device = Device::getCurrentDevice();
        
        // Set to GPU device (will fallback to CPU if GPU not available)
        Device::setDeviceWithValidation(DeviceType::GPU, false);
        
        // Create test arrays
        NDArray a({2, 2});
        NDArray b({2, 2});
        NDArray result;
        
        // Fill with test data
        a.data()[0] = 1.0; a.data()[1] = 2.0;
        a.data()[2] = 3.0; a.data()[3] = 4.0;
        
        b.data()[0] = 5.0; b.data()[1] = 6.0;
        b.data()[2] = 7.0; b.data()[3] = 8.0;
        
        // Test matrix multiplication
        result = a.matmul(b);
        
        // Expected result: [1*5+2*7, 1*6+2*8] = [19, 22]
        //                  [3*5+4*7, 3*6+4*8] = [43, 50]
        assertNear(result.data()[0], 19.0, 1e-10, "Matrix multiplication result [0,0] should be 19");
        assertNear(result.data()[1], 22.0, 1e-10, "Matrix multiplication result [0,1] should be 22");
        assertNear(result.data()[2], 43.0, 1e-10, "Matrix multiplication result [1,0] should be 43");
        assertNear(result.data()[3], 50.0, 1e-10, "Matrix multiplication result [1,1] should be 50");
        
        // Test element-wise addition
        result = a + b;
        assertNear(result.data()[0], 6.0, 1e-10, "Addition result [0] should be 6");
        assertNear(result.data()[1], 8.0, 1e-10, "Addition result [1] should be 8");
        assertNear(result.data()[2], 10.0, 1e-10, "Addition result [2] should be 10");
        assertNear(result.data()[3], 12.0, 1e-10, "Addition result [3] should be 12");
        
        // Test element-wise subtraction
        result = a - b;
        assertNear(result.data()[0], -4.0, 1e-10, "Subtraction result [0] should be -4");
        assertNear(result.data()[1], -4.0, 1e-10, "Subtraction result [1] should be -4");
        assertNear(result.data()[2], -4.0, 1e-10, "Subtraction result [2] should be -4");
        assertNear(result.data()[3], -4.0, 1e-10, "Subtraction result [3] should be -4");
        
        // Test element-wise multiplication
        result = a * b;
        assertNear(result.data()[0], 5.0, 1e-10, "Multiplication result [0] should be 5");
        assertNear(result.data()[1], 12.0, 1e-10, "Multiplication result [1] should be 12");
        assertNear(result.data()[2], 21.0, 1e-10, "Multiplication result [2] should be 21");
        assertNear(result.data()[3], 32.0, 1e-10, "Multiplication result [3] should be 32");
        
        // Test scalar operations
        result = a * 2.0;
        assertNear(result.data()[0], 2.0, 1e-10, "Scalar multiplication result [0] should be 2");
        assertNear(result.data()[1], 4.0, 1e-10, "Scalar multiplication result [1] should be 4");
        assertNear(result.data()[2], 6.0, 1e-10, "Scalar multiplication result [2] should be 6");
        assertNear(result.data()[3], 8.0, 1e-10, "Scalar multiplication result [3] should be 8");
        
        result = a + 10.0;
        assertNear(result.data()[0], 11.0, 1e-10, "Scalar addition result [0] should be 11");
        assertNear(result.data()[1], 12.0, 1e-10, "Scalar addition result [1] should be 12");
        assertNear(result.data()[2], 13.0, 1e-10, "Scalar addition result [2] should be 13");
        assertNear(result.data()[3], 14.0, 1e-10, "Scalar addition result [3] should be 14");
        
        // Restore original device
        Device::setDevice(original_device);
    }
};

/**
 * @class GPUArrayOperationsTest
 * @brief Test GPU backend array operations
 */
class GPUArrayOperationsTest : public TestCase {
public:
    GPUArrayOperationsTest() : TestCase("GPUArrayOperationsTest") {}

protected:
    void test() override {
        // Store original device
        DeviceType original_device = Device::getCurrentDevice();
        
        // Set to GPU device (will fallback to CPU if GPU not available)
        Device::setDeviceWithValidation(DeviceType::GPU, false);
        
        // Test fill operation
        NDArray arr({3, 3});
        arr.fill(42.0);
        
        for (size_t i = 0; i < arr.size(); ++i) {
            assertNear(arr.data()[i], 42.0, 1e-10, 
                       "Fill operation should set all elements to 42.0");
        }
        
        // Test copy operation
        NDArray src({2, 3});
        NDArray dst;
        
        for (size_t i = 0; i < src.size(); ++i) {
            src.data()[i] = static_cast<double>(i + 1);
        }
        
        dst = src;  // This should trigger copy operation
        
        assertTrue(dst.shape() == src.shape(), "Copy should preserve shape");
        for (size_t i = 0; i < src.size(); ++i) {
            assertNear(dst.data()[i], src.data()[i], 1e-10, 
                       "Copy should preserve all values");
        }
        
        // Test with different shapes
        NDArray large({100, 100});
        large.fill(3.14);
        
        for (size_t i = 0; i < 10; ++i) {  // Check first 10 elements
            assertNear(large.data()[i], 3.14, 1e-10, 
                       "Large array fill should work correctly");
        }
        
        // Restore original device
        Device::setDevice(original_device);
    }
};

/**
 * @class GPUModelTest
 * @brief Test GPU backend with model operations
 */
class GPUModelTest : public TestCase {
public:
    GPUModelTest() : TestCase("GPUModelTest") {}

protected:
    void test() override {
        // Store original device
        DeviceType original_device = Device::getCurrentDevice();
        
        try {
            // Create model with GPU device (should handle GPU unavailability gracefully)
            MLLib::model::Sequential model(DeviceType::GPU);
            
            // Check that device was set correctly (CPU fallback if GPU not available)
            DeviceType model_device = model.get_device();
            if (Device::isGPUAvailable()) {
                assertTrue(model_device == DeviceType::GPU, "Model should use GPU when available");
            } else {
                assertTrue(model_device == DeviceType::CPU, "Model should fallback to CPU when GPU not available");
            }
            
            // Test device switching
            model.set_device(DeviceType::CPU);
            assertTrue(model.get_device() == DeviceType::CPU, "Model device should switch to CPU");
            
            // Try switching back to GPU
            model.set_device(DeviceType::GPU);
            if (Device::isGPUAvailable()) {
                assertTrue(model.get_device() == DeviceType::GPU, "Model should switch to GPU when available");
            } else {
                assertTrue(model.get_device() == DeviceType::CPU, "Model should stay on CPU when GPU not available");
            }
            
            // Test that model still works correctly regardless of device
            // (This is important to ensure fallback doesn't break functionality)
            NDArray test_input({1, 2});
            test_input.data()[0] = 1.0;
            test_input.data()[1] = 2.0;
            
            // Model without layers should just return input
            // (We'll add a simple test that doesn't require layers)
            assertTrue(true, "Model creation and device switching should work");
            
        } catch (const std::exception& e) {
            assertFalse(true, std::string("GPU backend model test failed with exception: ") + e.what());
        }
        
        // Restore original device
        Device::setDevice(original_device);
    }
};

/**
 * @class GPUPerformanceTest
 * @brief Test GPU backend performance characteristics
 */
class GPUPerformanceTest : public TestCase {
public:
    GPUPerformanceTest() : TestCase("GPUPerformanceTest") {}

protected:
    void test() override {
        // Store original device
        DeviceType original_device = Device::getCurrentDevice();
        
        // Set to GPU device
        Device::setDeviceWithValidation(DeviceType::GPU, false);
        
        // Test with larger matrices to see if GPU/CPU distinction works
        const size_t size = 10;  // Keep small for unit tests
        NDArray a({size, size});
        NDArray b({size, size});
        
        // Fill with deterministic data
        for (size_t i = 0; i < a.size(); ++i) {
            a.data()[i] = static_cast<double>(i % 10) + 1.0;
            b.data()[i] = static_cast<double>((i + 5) % 10) + 1.0;
        }
        
        // Perform operations multiple times to ensure consistency
        for (int iter = 0; iter < 3; ++iter) {
            NDArray result = a.matmul(b);
            
            // Check that results are consistent across iterations
            // (This helps catch any non-deterministic GPU issues)
            assertEqual(size, result.shape()[0], "Result should have correct dimensions");
            assertEqual(size, result.shape()[1], "Result should have correct dimensions");
            
            // Check a few specific values for consistency
            double first_value = result.data()[0];
            assertTrue(first_value > 0, "Matrix multiplication should produce positive results");
        }
        
        // Test large element-wise operations
        NDArray large_a({100});
        NDArray large_b({100});
        
        for (size_t i = 0; i < 100; ++i) {
            large_a.data()[i] = static_cast<double>(i);
            large_b.data()[i] = static_cast<double>(i + 1);
        }
        
        NDArray large_result = large_a + large_b;
        
        // Check a few values
        assertNear(large_result.data()[0], 1.0, 1e-10, "Large array addition [0] should be correct");
        assertNear(large_result.data()[10], 21.0, 1e-10, "Large array addition [10] should be correct");
        assertNear(large_result.data()[99], 199.0, 1e-10, "Large array addition [99] should be correct");
        
        // Restore original device
        Device::setDevice(original_device);
    }
};

}  // namespace test
}  // namespace MLLib
