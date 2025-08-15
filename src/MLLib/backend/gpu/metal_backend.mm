/**
 * @file metal_backend.mm
 * @brief Apple Metal backend implementation (Objective-C++)
 */

#ifdef WITH_METAL

#include "../../../../include/MLLib/backend/metal_backend.hpp"
#include <iostream>
#include <stdexcept>

namespace MLLib {
namespace Backend {

id<MTLDevice> MetalBackend::device_ = nil;
id<MTLCommandQueue> MetalBackend::command_queue_ = nil;
id<MTLLibrary> MetalBackend::library_ = nil;
bool MetalBackend::initialized_ = false;

id<MTLComputePipelineState> MetalBackend::relu_pipeline_ = nil;
id<MTLComputePipelineState> MetalBackend::sigmoid_pipeline_ = nil;
id<MTLComputePipelineState> MetalBackend::tanh_pipeline_ = nil;

bool MetalBackend::isAvailable() {
    return MTLCreateSystemDefaultDevice() != nil;
}

void MetalBackend::initialize() {
    if (initialized_) return;
    
    @autoreleasepool {
        device_ = MTLCreateSystemDefaultDevice();
        if (!device_) {
            throw std::runtime_error("Metal device not available");
        }
        
        command_queue_ = [device_ newCommandQueue];
        if (!command_queue_) {
            throw std::runtime_error("Failed to create Metal command queue");
        }
        
        // Create default Metal library
        library_ = [device_ newDefaultLibrary];
        if (!library_) {
            throw std::runtime_error("Failed to create Metal library");
        }
        
        initializeKernels();
        
        initialized_ = true;
        std::cout << "Metal backend initialized successfully" << std::endl;
        std::cout << "Device: " << device_.name.UTF8String << std::endl;
    }
}

void MetalBackend::cleanup() {
    if (!initialized_) return;
    
    @autoreleasepool {
        if (relu_pipeline_) {
            [relu_pipeline_ release];
            relu_pipeline_ = nil;
        }
        if (sigmoid_pipeline_) {
            [sigmoid_pipeline_ release];
            sigmoid_pipeline_ = nil;
        }
        if (tanh_pipeline_) {
            [tanh_pipeline_ release];
            tanh_pipeline_ = nil;
        }
        if (library_) {
            [library_ release];
            library_ = nil;
        }
        if (command_queue_) {
            [command_queue_ release];
            command_queue_ = nil;
        }
        if (device_) {
            [device_ release];
            device_ = nil;
        }
    }
    
    initialized_ = false;
}

void MetalBackend::initializeKernels() {
    @autoreleasepool {
        NSError* error = nil;
        
        // Metal Shading Language kernel source
        NSString* kernelSource = @R"(
            #include <metal_stdlib>
            using namespace metal;
            
            kernel void relu_kernel(device const float* input [[buffer(0)]],
                                   device float* output [[buffer(1)]],
                                   uint index [[thread_position_in_grid]]) {
                output[index] = max(0.0f, input[index]);
            }
            
            kernel void sigmoid_kernel(device const float* input [[buffer(0)]],
                                      device float* output [[buffer(1)]],
                                      uint index [[thread_position_in_grid]]) {
                output[index] = 1.0f / (1.0f + exp(-input[index]));
            }
            
            kernel void tanh_kernel(device const float* input [[buffer(0)]],
                                   device float* output [[buffer(1)]],
                                   uint index [[thread_position_in_grid]]) {
                output[index] = tanh(input[index]);
            }
        )";
        
        id<MTLLibrary> kernelLibrary = [device_ newLibraryWithSource:kernelSource
                                                             options:nil
                                                               error:&error];
        if (!kernelLibrary) {
            throw std::runtime_error("Failed to compile Metal kernels");
        }
        
        // Create compute pipeline states
        id<MTLFunction> reluFunction = [kernelLibrary newFunctionWithName:@"relu_kernel"];
        relu_pipeline_ = [device_ newComputePipelineStateWithFunction:reluFunction error:&error];
        
        id<MTLFunction> sigmoidFunction = [kernelLibrary newFunctionWithName:@"sigmoid_kernel"];
        sigmoid_pipeline_ = [device_ newComputePipelineStateWithFunction:sigmoidFunction error:&error];
        
        id<MTLFunction> tanhFunction = [kernelLibrary newFunctionWithName:@"tanh_kernel"];
        tanh_pipeline_ = [device_ newComputePipelineStateWithFunction:tanhFunction error:&error];
        
        if (!relu_pipeline_ || !sigmoid_pipeline_ || !tanh_pipeline_) {
            throw std::runtime_error("Failed to create Metal compute pipeline states");
        }
        
        [kernelLibrary release];
        [reluFunction release];
        [sigmoidFunction release];
        [tanhFunction release];
    }
}

void* MetalBackend::allocateMemory(size_t size) {
    if (!device_) {
        throw std::runtime_error("Metal backend not initialized");
    }
    
    @autoreleasepool {
        id<MTLBuffer> buffer = [device_ newBufferWithLength:size options:MTLResourceStorageModeShared];
        if (!buffer) {
            throw std::runtime_error("Failed to allocate Metal buffer");
        }
        return (__bridge_retained void*)buffer;
    }
}

void MetalBackend::deallocateMemory(void* ptr) {
    if (ptr) {
        @autoreleasepool {
            id<MTLBuffer> buffer = (__bridge_transfer id<MTLBuffer>)ptr;
            [buffer release];
        }
    }
}

void MetalBackend::copyToDevice(void* dst, const void* src, size_t size) {
    @autoreleasepool {
        id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)dst;
        memcpy(buffer.contents, src, size);
    }
}

void MetalBackend::copyFromDevice(void* dst, const void* src, size_t size) {
    @autoreleasepool {
        id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)src;
        memcpy(dst, buffer.contents, size);
    }
}

void MetalBackend::copyDeviceToDevice(void* dst, const void* src, size_t size) {
    @autoreleasepool {
        id<MTLBuffer> dstBuffer = (__bridge id<MTLBuffer>)dst;
        id<MTLBuffer> srcBuffer = (__bridge id<MTLBuffer>)src;
        memcpy(dstBuffer.contents, srcBuffer.contents, size);
    }
}

void MetalBackend::gemm(bool transposeA, bool transposeB,
                        int m, int n, int k,
                        double alpha, const double* A, int lda,
                        const double* B, int ldb,
                        double beta, double* C, int ldc) {
    
    // Note: For simplicity, this is a basic CPU implementation
    // In a production system, you would use Metal Performance Shaders (MPS)
    // or implement custom Metal kernels for GEMM operations
    
    // Convert to float for Metal (Metal prefers float over double)
    // This is a simplified implementation - production code would handle this properly
    
    throw std::runtime_error("Metal GEMM not yet implemented - use MPS for production");
}

void MetalBackend::relu(const double* input, double* output, size_t size) {
    if (!device_ || !relu_pipeline_) {
        throw std::runtime_error("Metal backend not initialized");
    }
    
    @autoreleasepool {
        id<MTLCommandBuffer> commandBuffer = [command_queue_ commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        
        // Note: This is simplified - production code would handle double->float conversion
        // For now, assuming float data
        id<MTLBuffer> inputBuffer = (__bridge id<MTLBuffer>)input;
        id<MTLBuffer> outputBuffer = (__bridge id<MTLBuffer>)output;
        
        [encoder setComputePipelineState:relu_pipeline_];
        [encoder setBuffer:inputBuffer offset:0 atIndex:0];
        [encoder setBuffer:outputBuffer offset:0 atIndex:1];
        
        MTLSize threadsPerGroup = MTLSizeMake(256, 1, 1);
        MTLSize numThreadgroups = MTLSizeMake((size + 255) / 256, 1, 1);
        
        [encoder dispatchThreadgroups:numThreadgroups threadsPerThreadgroup:threadsPerGroup];
        [encoder endEncoding];
        
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
    }
}

void MetalBackend::sigmoid(const double* input, double* output, size_t size) {
    if (!device_ || !sigmoid_pipeline_) {
        throw std::runtime_error("Metal backend not initialized");
    }
    
    @autoreleasepool {
        id<MTLCommandBuffer> commandBuffer = [command_queue_ commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        
        id<MTLBuffer> inputBuffer = (__bridge id<MTLBuffer>)input;
        id<MTLBuffer> outputBuffer = (__bridge id<MTLBuffer>)output;
        
        [encoder setComputePipelineState:sigmoid_pipeline_];
        [encoder setBuffer:inputBuffer offset:0 atIndex:0];
        [encoder setBuffer:outputBuffer offset:0 atIndex:1];
        
        MTLSize threadsPerGroup = MTLSizeMake(256, 1, 1);
        MTLSize numThreadgroups = MTLSizeMake((size + 255) / 256, 1, 1);
        
        [encoder dispatchThreadgroups:numThreadgroups threadsPerThreadgroup:threadsPerGroup];
        [encoder endEncoding];
        
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
    }
}

void MetalBackend::tanh_activation(const double* input, double* output, size_t size) {
    if (!device_ || !tanh_pipeline_) {
        throw std::runtime_error("Metal backend not initialized");
    }
    
    @autoreleasepool {
        id<MTLCommandBuffer> commandBuffer = [command_queue_ commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        
        id<MTLBuffer> inputBuffer = (__bridge id<MTLBuffer>)input;
        id<MTLBuffer> outputBuffer = (__bridge id<MTLBuffer>)output;
        
        [encoder setComputePipelineState:tanh_pipeline_];
        [encoder setBuffer:inputBuffer offset:0 atIndex:0];
        [encoder setBuffer:outputBuffer offset:0 atIndex:1];
        
        MTLSize threadsPerGroup = MTLSizeMake(256, 1, 1);
        MTLSize numThreadgroups = MTLSizeMake((size + 255) / 256, 1, 1);
        
        [encoder dispatchThreadgroups:numThreadgroups threadsPerThreadgroup:threadsPerGroup];
        [encoder endEncoding];
        
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
    }
}

void MetalBackend::synchronize() {
    // Metal operations are already synchronous in this implementation
    // In an async implementation, you would wait for command buffers here
}

int MetalBackend::getDeviceCount() {
    // macOS typically has one integrated GPU
    return isAvailable() ? 1 : 0;
}

void MetalBackend::setDevice(int device) {
    if (device != 0) {
        throw std::runtime_error("Only device 0 available on Metal");
    }
    // No-op since we only have one device
}

std::string MetalBackend::getDeviceName(int device) {
    if (device != 0) {
        return "Invalid device";
    }
    
    if (device_) {
        return std::string(device_.name.UTF8String);
    }
    
    @autoreleasepool {
        id<MTLDevice> tempDevice = MTLCreateSystemDefaultDevice();
        if (tempDevice) {
            std::string name = std::string(tempDevice.name.UTF8String);
            [tempDevice release];
            return name;
        }
    }
    
    return "Unknown Apple GPU";
}

} // namespace Backend
} // namespace MLLib

#endif // WITH_METAL
