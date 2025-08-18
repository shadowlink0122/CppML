/**
 * @file gpu_kernel_manager.mm
 * @brief Generic GPU kernel management implementation
 */

#include "../../../include/MLLib/backend/gpu_kernel_manager.hpp"
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <chrono>

#ifdef WITH_METAL
#import <Metal/Metal.h>
#endif

namespace MLLib {
namespace Backend {

// Static member initialization
bool GPUKernelManager::initialized_ = false;

#ifdef WITH_METAL
std::unordered_map<std::string, GPUKernelManager::MetalKernel> GPUKernelManager::metal_kernels_;
id<MTLDevice> GPUKernelManager::device_ = nil;
id<MTLCommandQueue> GPUKernelManager::command_queue_ = nil;
#endif

std::unordered_map<std::string, ActivationKernelRegistry::ActivationDef> ActivationKernelRegistry::activations_;

// GPUKernelManager Implementation
void GPUKernelManager::executeUnaryKernel(
    const std::string& kernel_name,
    const double* input, 
    double* output, 
    size_t size,
    const std::vector<double>& params) {
    
    if (!initialized_) {
        initializeBuiltinKernels();
    }

#ifdef WITH_METAL
    @autoreleasepool {
        // Convert input to float
        auto input_float = convertToFloat(input, size);
        size_t bufferSize = size * sizeof(float);
        
        // Create buffers
        id<MTLBuffer> inputBuffer = [device_ newBufferWithBytes:input_float.data()
                                                        length:bufferSize
                                                       options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> outputBuffer = [device_ newBufferWithLength:bufferSize
                                                          options:MTLResourceStorageModeShared];
        
        std::vector<id<MTLBuffer>> buffers = {inputBuffer, outputBuffer};
        
        // Add parameter buffers if needed
        std::vector<id<MTLBuffer>> paramBuffers;
        for (const auto& param : params) {
            float paramFloat = static_cast<float>(param);
            id<MTLBuffer> paramBuffer = [device_ newBufferWithBytes:&paramFloat
                                                            length:sizeof(float)
                                                           options:MTLResourceStorageModeShared];
            paramBuffers.push_back(paramBuffer);
            buffers.push_back(paramBuffer);
        }
        
        // Execute kernel
        executeMetalKernel(kernel_name, buffers, size);
        
        // Convert result back to double
        float* result = (float*)[outputBuffer contents];
        convertFromFloat(result, output, size);
    }
#else
    // CPU fallback
    throw std::runtime_error("GPU kernels not available in this build");
#endif
}

void GPUKernelManager::executeBinaryKernel(
    const std::string& kernel_name,
    const double* input1,
    const double* input2, 
    double* output,
    size_t size,
    const std::vector<double>& params) {
    
    if (!initialized_) {
        initializeBuiltinKernels();
    }

#ifdef WITH_METAL
    @autoreleasepool {
        // Convert inputs to float
        auto input1_float = convertToFloat(input1, size);
        auto input2_float = convertToFloat(input2, size);
        size_t bufferSize = size * sizeof(float);
        
        // Create buffers
        id<MTLBuffer> input1Buffer = [device_ newBufferWithBytes:input1_float.data()
                                                         length:bufferSize
                                                        options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> input2Buffer = [device_ newBufferWithBytes:input2_float.data()
                                                         length:bufferSize
                                                        options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> outputBuffer = [device_ newBufferWithLength:bufferSize
                                                          options:MTLResourceStorageModeShared];
        
        std::vector<id<MTLBuffer>> buffers = {input1Buffer, input2Buffer, outputBuffer};
        
        // Add parameter buffers if needed
        for (const auto& param : params) {
            float paramFloat = static_cast<float>(param);
            id<MTLBuffer> paramBuffer = [device_ newBufferWithBytes:&paramFloat
                                                            length:sizeof(float)
                                                           options:MTLResourceStorageModeShared];
            buffers.push_back(paramBuffer);
        }
        
        // Execute kernel
        executeMetalKernel(kernel_name, buffers, size);
        
        // Convert result back to double
        float* result = (float*)[outputBuffer contents];
        convertFromFloat(result, output, size);
    }
#else
    // CPU fallback
    throw std::runtime_error("GPU kernels not available in this build");
#endif
}

void GPUKernelManager::registerKernel(const KernelParams& kernel_params) {
#ifdef WITH_METAL
    compileMetalKernel(kernel_params.name, kernel_params.source, kernel_params.constants.size());
#endif
}

void GPUKernelManager::initializeBuiltinKernels() {
    if (initialized_) return;
    
#ifdef WITH_METAL
    // Initialize Metal device and command queue
    device_ = MTLCreateSystemDefaultDevice();
    if (!device_) {
        throw std::runtime_error("Metal device not available");
    }
    command_queue_ = [device_ newCommandQueue];
    if (!command_queue_) {
        throw std::runtime_error("Failed to create Metal command queue");
    }
#endif

    // Initialize activation functions
    ActivationKernelRegistry::initializeBuiltinActivations();
    
    initialized_ = true;
}

#ifdef WITH_METAL
void GPUKernelManager::compileMetalKernel(const std::string& name, const std::string& source, size_t param_count) {
    @autoreleasepool {
        NSString* nsSource = [NSString stringWithUTF8String:source.c_str()];
        NSError* error = nil;
        
        id<MTLLibrary> library = [device_ newLibraryWithSource:nsSource options:nil error:&error];
        if (!library) {
            throw std::runtime_error("Failed to compile kernel: " + name + 
                                   " Error: " + std::string([[error description] UTF8String]));
        }
        
        NSString* kernelName = [NSString stringWithUTF8String:(name + "_kernel").c_str()];
        id<MTLFunction> function = [library newFunctionWithName:kernelName];
        if (!function) {
            throw std::runtime_error("Failed to find function: " + name + "_kernel");
        }
        
        id<MTLComputePipelineState> pipeline = [device_ newComputePipelineStateWithFunction:function error:&error];
        if (!pipeline) {
            throw std::runtime_error("Failed to create pipeline for: " + name +
                                   " Error: " + std::string([[error description] UTF8String]));
        }
        
        MetalKernel kernel = {pipeline, source, param_count};
        metal_kernels_[name] = kernel;
    }
}

void GPUKernelManager::executeMetalKernel(const std::string& name, const std::vector<id<MTLBuffer>>& buffers, size_t size) {
    auto it = metal_kernels_.find(name);
    if (it == metal_kernels_.end()) {
        throw std::runtime_error("Kernel not found: " + name);
    }
    
    @autoreleasepool {
        id<MTLCommandBuffer> commandBuffer = [command_queue_ commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        
        [encoder setComputePipelineState:it->second.pipeline];
        
        // Set buffers
        for (size_t i = 0; i < buffers.size(); ++i) {
            [encoder setBuffer:buffers[i] offset:0 atIndex:i];
        }
        
        // Calculate thread configuration
        MTLSize threadgroupSize = MTLSizeMake(256, 1, 1);
        MTLSize numThreadgroups = MTLSizeMake((size + 255) / 256, 1, 1);
        
        [encoder dispatchThreadgroups:numThreadgroups threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];
        
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
    }
}
#endif

std::vector<float> GPUKernelManager::convertToFloat(const double* data, size_t size) {
    std::vector<float> result(size);
    for (size_t i = 0; i < size; ++i) {
        result[i] = static_cast<float>(data[i]);
    }
    return result;
}

void GPUKernelManager::convertFromFloat(const float* data, double* output, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        output[i] = static_cast<double>(data[i]);
    }
}

void GPUKernelManager::cleanup() {
#ifdef WITH_METAL
    metal_kernels_.clear();
    device_ = nil;
    command_queue_ = nil;
#endif
    initialized_ = false;
}

// ActivationKernelRegistry Implementation
void ActivationKernelRegistry::registerActivation(const ActivationDef& def) {
    activations_[def.name] = def;
    
    // Generate and compile kernel
    std::string source = generateKernelSource(def.name, def.gpu_expression, def.param_names);
    KernelParams params = {def.name, source, {}};
    GPUKernelManager::registerKernel(params);
}

void ActivationKernelRegistry::executeActivation(
    const std::string& name,
    const double* input,
    double* output, 
    size_t size,
    const std::vector<double>& params) {
    
    auto it = activations_.find(name);
    if (it == activations_.end()) {
        throw std::runtime_error("Activation function not found: " + name);
    }
    
    if (it->second.has_parameters && params.size() != it->second.param_names.size()) {
        throw std::runtime_error("Parameter count mismatch for activation: " + name);
    }
    
    GPUKernelManager::executeUnaryKernel(name, input, output, size, params);
}

void ActivationKernelRegistry::initializeBuiltinActivations() {
    // ReLU
    registerActivation({
        "relu",
        "max(0.0f, input)",
        {},
        false
    });
    
    // Sigmoid  
    registerActivation({
        "sigmoid",
        "1.0f / (1.0f + exp(-input))",
        {},
        false
    });
    
    // Tanh
    registerActivation({
        "tanh",
        "tanh(input)",
        {},
        false
    });
    
    // LeakyReLU
    registerActivation({
        "leaky_relu", 
        "input > 0.0f ? input : alpha * input",
        {"alpha"},
        true
    });
    
    // GELU
    registerActivation({
        "gelu",
        "0.5f * input * (1.0f + tanh(0.7978845608f * (input + 0.044715f * input * input * input)))",
        {},
        false
    });
    
    // GELU Approximate
    registerActivation({
        "gelu_approx",
        "0.5f * input * (1.0f + tanh(0.7978845608f * (input + 0.044715f * input * input * input)))",
        {},
        false
    });
    
    // Softmax (requires special handling for normalization)
    registerActivation({
        "softmax_element",
        "exp(input - max_val) / sum_exp",  // Will need special 2-pass implementation
        {"max_val", "sum_exp"},
        true
    });
    
    // ELU
    registerActivation({
        "elu",
        "input > 0.0f ? input : alpha * (exp(input) - 1.0f)",
        {"alpha"},
        true
    });
    
    // Swish
    registerActivation({
        "swish",
        "input / (1.0f + exp(-input))",
        {},
        false
    });
}

std::string ActivationKernelRegistry::generateKernelSource(
    const std::string& name,
    const std::string& expression, 
    const std::vector<std::string>& param_names) {
    
    std::ostringstream oss;
    oss << R"(
#include <metal_stdlib>
using namespace metal;

kernel void )" << name << R"(_kernel(device const float* input [[buffer(0)]],
                       device float* output [[buffer(1)]],)";
    
    // Add parameter buffers
    for (size_t i = 0; i < param_names.size(); ++i) {
        oss << "\n                       device const float* " << param_names[i] 
            << " [[buffer(" << (i + 2) << ")]],";
    }
    
    oss << R"(
                       uint index [[thread_position_in_grid]]) {
    )";
    
    // Add parameter declarations
    for (const auto& param_name : param_names) {
        oss << "    float " << param_name << " = " << param_name << "[0];\n";
    }
    
    oss << "    output[index] = " << expression << ";\n";
    oss << "}\n";
    
    return oss.str();
}

} // namespace Backend
} // namespace MLLib
