/**
 * @file metal_backend.mm
 * @brief Apple Metal backend implementation (Objective-C++)
 */

#ifdef WITH_METAL

#ifdef CI
// CI environment: Provide stub implementations to avoid linking issues
#include "../../../../include/MLLib/backend/metal_backend.hpp"

namespace MLLib {
namespace Backend {

// Stub static variables
id<MTLDevice> MetalBackend::device_ = nil;
id<MTLCommandQueue> MetalBackend::command_queue_ = nil;  
id<MTLLibrary> MetalBackend::library_ = nil;
bool MetalBackend::initialized_ = false;

id<MTLComputePipelineState> MetalBackend::relu_pipeline_ = nil;
id<MTLComputePipelineState> MetalBackend::sigmoid_pipeline_ = nil;
id<MTLComputePipelineState> MetalBackend::tanh_pipeline_ = nil;

// Stub implementations
bool MetalBackend::isAvailable() { return false; }
void MetalBackend::initialize() {}
void MetalBackend::cleanup() {}
void MetalBackend::initializeKernels() {}
void* MetalBackend::allocateMemory(size_t) { return nullptr; }
void MetalBackend::deallocateMemory(void*) {}
void MetalBackend::copyToDevice(void*, const void*, size_t) {}
void MetalBackend::copyFromDevice(void*, const void*, size_t) {}  
void MetalBackend::copyDeviceToDevice(void*, const void*, size_t) {}
void MetalBackend::gemm(bool, bool, int, int, int, double, const double*, int, const double*, int, double, double*, int) {}
void MetalBackend::matmul(const double*, const double*, double*, int, int, int) {}
void MetalBackend::relu(const double*, double*, size_t) {}
void MetalBackend::sigmoid(const double*, double*, size_t) {}
void MetalBackend::tanh_activation(const double*, double*, size_t) {}

} // namespace Backend
} // namespace MLLib

#else
// Normal Metal implementation
#include "../../../../include/MLLib/backend/metal_backend.hpp"
#include <iostream>
#include <stdexcept>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

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
        
        // Create default Metal library - use system library for MPS
        library_ = [device_ newDefaultLibrary];
        if (!library_) {
            // Try to create a minimal library if default fails
            NSString *kernelSource = @"#include <metal_stdlib>\nusing namespace metal;\nkernel void dummy() {}";
            NSError* error = nil;
            library_ = [device_ newLibraryWithSource:kernelSource options:nil error:&error];
            if (!library_) {
                NSString* errorDesc = error ? error.localizedDescription : @"Unknown error";
                printf("Warning: Could not create Metal library: %s\n", [errorDesc UTF8String]);
                // Continue without library for MPS usage
            }
        }
        
        // Skip kernel initialization for now - use runtime compilation
        // initializeKernels();
        
        initialized_ = true;
        printf("Metal backend initialized successfully (runtime compilation mode)\n");
        printf("Device: %s\n", device_.name.UTF8String);
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
            
            kernel void relu_kernel(device const double* input [[buffer(0)]],
                                   device double* output [[buffer(1)]],
                                   uint index [[thread_position_in_grid]]) {
                output[index] = max(0.0, input[index]);
            }
            
            kernel void sigmoid_kernel(device const double* input [[buffer(0)]],
                                      device double* output [[buffer(1)]],
                                      uint index [[thread_position_in_grid]]) {
                output[index] = 1.0 / (1.0 + exp(-input[index]));
            }
            
            kernel void tanh_kernel(device const double* input [[buffer(0)]],
                                   device double* output [[buffer(1)]],
                                   uint index [[thread_position_in_grid]]) {
                output[index] = tanh(input[index]);
            }
        )";
        
        id<MTLLibrary> kernelLibrary = [device_ newLibraryWithSource:kernelSource
                                                             options:nil
                                                               error:&error];
        if (!kernelLibrary) {
            NSString* errorDesc = error ? error.localizedDescription : @"Unknown error";
            printf("Metal kernel compilation error: %s\n", [errorDesc UTF8String]);
            throw std::runtime_error("Failed to compile Metal kernels: " + std::string([errorDesc UTF8String]));
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
    
    // Use the simpler matmul interface for now
    matmul(A, B, C, m, n, k);
}

void MetalBackend::matmul(const double* A, const double* B, double* C, 
                         int m, int n, int k) {
  if (!initialized_) {
    initialize();
  }

  // Use Metal Performance Shaders for maximum performance
  // Convert double to float for Metal Performance Shaders compatibility
  std::vector<float> A_float(m * k);
  std::vector<float> B_float(k * n);
  
  // Convert input data to float
  for (int i = 0; i < m * k; ++i) {
    A_float[i] = static_cast<float>(A[i]);
  }
  for (int i = 0; i < k * n; ++i) {
    B_float[i] = static_cast<float>(B[i]);
  }
  
  size_t bufferSizeA = m * k * sizeof(float);
  size_t bufferSizeB = k * n * sizeof(float);  
  size_t bufferSizeC = m * n * sizeof(float);

  id<MTLBuffer> bufferA = [device_ newBufferWithBytes:A_float.data()
                                              length:bufferSizeA
                                             options:MTLResourceStorageModeShared];
  
  id<MTLBuffer> bufferB = [device_ newBufferWithBytes:B_float.data()
                                              length:bufferSizeB
                                             options:MTLResourceStorageModeShared];
  
  id<MTLBuffer> bufferC = [device_ newBufferWithLength:bufferSizeC
                                               options:MTLResourceStorageModeShared];

  // Use Metal Performance Shaders for optimal performance
  printf("DEBUG: Starting MPS matrix multiplication\n");
  
  @try {
    MPSMatrixDescriptor* descA = [MPSMatrixDescriptor 
                                        matrixDescriptorWithRows:m 
                                                         columns:k 
                                                        matrices:1 
                                                        rowBytes:k * sizeof(float)
                                                        matrixBytes:m * k * sizeof(float)
                                                        dataType:MPSDataTypeFloat32];

    MPSMatrixDescriptor* descB = [MPSMatrixDescriptor 
                                        matrixDescriptorWithRows:k 
                                                         columns:n 
                                                        matrices:1 
                                                        rowBytes:n * sizeof(float)
                                                        matrixBytes:k * n * sizeof(float)
                                                        dataType:MPSDataTypeFloat32];

    MPSMatrixDescriptor* descC = [MPSMatrixDescriptor 
                                        matrixDescriptorWithRows:m 
                                                         columns:n 
                                                        matrices:1 
                                                        rowBytes:n * sizeof(float)
                                                        matrixBytes:m * n * sizeof(float)
                                                        dataType:MPSDataTypeFloat32];

    MPSMatrix* matrixA = [[MPSMatrix alloc] initWithBuffer:bufferA descriptor:descA];
    MPSMatrix* matrixB = [[MPSMatrix alloc] initWithBuffer:bufferB descriptor:descB];
    MPSMatrix* matrixC = [[MPSMatrix alloc] initWithBuffer:bufferC descriptor:descC];

    MPSMatrixMultiplication* matmul = [[MPSMatrixMultiplication alloc] 
                                         initWithDevice:device_
                                         transposeLeft:NO
                                         transposeRight:NO
                                         resultRows:m
                                         resultColumns:n
                                         interiorColumns:k
                                         alpha:1.0
                                         beta:0.0];

    id<MTLCommandBuffer> commandBuffer = [command_queue_ commandBuffer];
    [matmul encodeToCommandBuffer:commandBuffer leftMatrix:matrixA rightMatrix:matrixB resultMatrix:matrixC];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];

    printf("DEBUG: MPS matrix multiplication completed\n");

    // Convert result back to double
    float* result_float = (float*)[bufferC contents];
    for (int i = 0; i < m * n; ++i) {
      C[i] = static_cast<double>(result_float[i]);
    }

    // Clean up
    [matrixA release];
    [matrixB release];
    [matrixC release];
    [matmul release];
    
    printf("DEBUG: Metal matmul completed successfully with MPS\n");
    
  } @catch (NSException* exception) {
    printf("DEBUG: MPS failed: %s\n", [exception.reason UTF8String]);
    throw std::runtime_error("Metal Performance Shaders failed: " + std::string([exception.reason UTF8String]));
  }
}

void MetalBackend::relu(const double* input, double* output, size_t size) {
    if (!initialized_) {
        initialize();
    }
    
    @autoreleasepool {
        // Create buffers
        size_t bufferSize = size * sizeof(float);
        
        // Convert double to float for Metal compatibility
        std::vector<float> input_float(size);
        for (size_t i = 0; i < size; ++i) {
            input_float[i] = static_cast<float>(input[i]);
        }
        
        id<MTLBuffer> inputBuffer = [device_ newBufferWithBytes:input_float.data()
                                                        length:bufferSize
                                                       options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> outputBuffer = [device_ newBufferWithLength:bufferSize
                                                          options:MTLResourceStorageModeShared];
        
        // Create compute command
        id<MTLCommandBuffer> commandBuffer = [command_queue_ commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        
        // Use runtime kernel compilation if pipeline not available
        if (!relu_pipeline_) {
            NSString* kernelSource = @R"(
                #include <metal_stdlib>
                using namespace metal;
                
                kernel void relu_kernel(device const float* input [[buffer(0)]],
                                       device float* output [[buffer(1)]],
                                       uint index [[thread_position_in_grid]]) {
                    output[index] = max(0.0f, input[index]);
                }
            )";
            
            NSError* error = nil;
            id<MTLLibrary> kernelLib = [device_ newLibraryWithSource:kernelSource options:nil error:&error];
            id<MTLFunction> function = [kernelLib newFunctionWithName:@"relu_kernel"];
            relu_pipeline_ = [device_ newComputePipelineStateWithFunction:function error:&error];
        }
        
        [encoder setComputePipelineState:relu_pipeline_];
        [encoder setBuffer:inputBuffer offset:0 atIndex:0];
        [encoder setBuffer:outputBuffer offset:0 atIndex:1];
        
        MTLSize threadgroupSize = MTLSizeMake(256, 1, 1);
        MTLSize numThreadgroups = MTLSizeMake((size + 255) / 256, 1, 1);
        
        [encoder dispatchThreadgroups:numThreadgroups threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];
        
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        // Convert result back to double
        float* result = (float*)[outputBuffer contents];
        for (size_t i = 0; i < size; ++i) {
            output[i] = static_cast<double>(result[i]);
        }
    }
}

void MetalBackend::sigmoid(const double* input, double* output, size_t size) {
    if (!initialized_) {
        initialize();
    }
    
    @autoreleasepool {
        // Create buffers
        size_t bufferSize = size * sizeof(float);
        
        // Convert double to float for Metal compatibility
        std::vector<float> input_float(size);
        for (size_t i = 0; i < size; ++i) {
            input_float[i] = static_cast<float>(input[i]);
        }
        
        id<MTLBuffer> inputBuffer = [device_ newBufferWithBytes:input_float.data()
                                                        length:bufferSize
                                                       options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> outputBuffer = [device_ newBufferWithLength:bufferSize
                                                          options:MTLResourceStorageModeShared];
        
        // Create compute command
        id<MTLCommandBuffer> commandBuffer = [command_queue_ commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        
        // Use runtime kernel compilation if pipeline not available
        if (!sigmoid_pipeline_) {
            NSString* kernelSource = @R"(
                #include <metal_stdlib>
                using namespace metal;
                
                kernel void sigmoid_kernel(device const float* input [[buffer(0)]],
                                          device float* output [[buffer(1)]],
                                          uint index [[thread_position_in_grid]]) {
                    output[index] = 1.0f / (1.0f + exp(-input[index]));
                }
            )";
            
            NSError* error = nil;
            id<MTLLibrary> kernelLib = [device_ newLibraryWithSource:kernelSource options:nil error:&error];
            id<MTLFunction> function = [kernelLib newFunctionWithName:@"sigmoid_kernel"];
            sigmoid_pipeline_ = [device_ newComputePipelineStateWithFunction:function error:&error];
        }
        
        [encoder setComputePipelineState:sigmoid_pipeline_];
        [encoder setBuffer:inputBuffer offset:0 atIndex:0];
        [encoder setBuffer:outputBuffer offset:0 atIndex:1];
        
        MTLSize threadgroupSize = MTLSizeMake(256, 1, 1);
        MTLSize numThreadgroups = MTLSizeMake((size + 255) / 256, 1, 1);
        
        [encoder dispatchThreadgroups:numThreadgroups threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];
        
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        // Convert result back to double
        float* result = (float*)[outputBuffer contents];
        for (size_t i = 0; i < size; ++i) {
            output[i] = static_cast<double>(result[i]);
        }
    }
}

void MetalBackend::tanh_activation(const double* input, double* output, size_t size) {
    if (!initialized_) {
        initialize();
    }
    
    @autoreleasepool {
        // Create buffers
        size_t bufferSize = size * sizeof(float);
        
        // Convert double to float for Metal compatibility
        std::vector<float> input_float(size);
        for (size_t i = 0; i < size; ++i) {
            input_float[i] = static_cast<float>(input[i]);
        }
        
        id<MTLBuffer> inputBuffer = [device_ newBufferWithBytes:input_float.data()
                                                        length:bufferSize
                                                       options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> outputBuffer = [device_ newBufferWithLength:bufferSize
                                                          options:MTLResourceStorageModeShared];
        
        // Create compute command
        id<MTLCommandBuffer> commandBuffer = [command_queue_ commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        
        // Use runtime kernel compilation if pipeline not available
        if (!tanh_pipeline_) {
            NSString* kernelSource = @R"(
                #include <metal_stdlib>
                using namespace metal;
                
                kernel void tanh_kernel(device const float* input [[buffer(0)]],
                                       device float* output [[buffer(1)]],
                                       uint index [[thread_position_in_grid]]) {
                    output[index] = tanh(input[index]);
                }
            )";
            
            NSError* error = nil;
            id<MTLLibrary> kernelLib = [device_ newLibraryWithSource:kernelSource options:nil error:&error];
            id<MTLFunction> function = [kernelLib newFunctionWithName:@"tanh_kernel"];
            tanh_pipeline_ = [device_ newComputePipelineStateWithFunction:function error:&error];
        }
        
        [encoder setComputePipelineState:tanh_pipeline_];
        [encoder setBuffer:inputBuffer offset:0 atIndex:0];
        [encoder setBuffer:outputBuffer offset:0 atIndex:1];
        
        MTLSize threadgroupSize = MTLSizeMake(256, 1, 1);
        MTLSize numThreadgroups = MTLSizeMake((size + 255) / 256, 1, 1);
        
        [encoder dispatchThreadgroups:numThreadgroups threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];
        
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        // Convert result back to double
        float* result = (float*)[outputBuffer contents];
        for (size_t i = 0; i < size; ++i) {
            output[i] = static_cast<double>(result[i]);
        }
    }
}

void MetalBackend::synchronize() {
    // Metal operations are already synchronous in this implementation
}

int MetalBackend::getDeviceCount() {
    return isAvailable() ? 1 : 0;
}

void MetalBackend::setDevice(int device) {
    if (device != 0) {
        throw std::runtime_error("Only device 0 available on Metal");
    }
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
            return name;
        }
    }
    
    return "Unknown Apple GPU";
}

} // namespace Backend
} // namespace MLLib

#endif // CI environment stub implementations

#endif // WITH_METAL
