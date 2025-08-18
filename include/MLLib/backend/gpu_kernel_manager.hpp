#pragma once

#include <string>
#include <unordered_map>
#include <memory>
#include <functional>

// Forward declarations instead of direct includes
#ifdef WITH_METAL
#ifdef __OBJC__
#import <Metal/Metal.h>
#else
// Forward declarations for C++ compilation
typedef void* id;
typedef void* MTLDevice;
typedef void* MTLCommandQueue;
typedef void* MTLBuffer;
typedef void* MTLComputePipelineState;
#endif
#endif

namespace MLLib {
namespace Backend {

/**
 * @brief GPU kernel parameter structure
 */
struct KernelParams {
    std::string name;
    std::string source;
    std::vector<double> constants;  // For parameterized kernels (e.g., LeakyReLU alpha)
};

/**
 * @brief Generic GPU kernel executor
 * Provides a unified interface for all activation functions and mathematical operations
 */
class GPUKernelManager {
public:
    /**
     * @brief Execute a unary operation (single input, single output)
     * @param kernel_name Name of the kernel to execute
     * @param input Input data
     * @param output Output data  
     * @param size Number of elements
     * @param params Optional parameters for parameterized kernels
     */
    static void executeUnaryKernel(
        const std::string& kernel_name,
        const double* input, 
        double* output, 
        size_t size,
        const std::vector<double>& params = {}
    );

    /**
     * @brief Execute a binary operation (two inputs, one output)
     * @param kernel_name Name of the kernel to execute
     * @param input1 First input data
     * @param input2 Second input data
     * @param output Output data
     * @param size Number of elements
     * @param params Optional parameters
     */
    static void executeBinaryKernel(
        const std::string& kernel_name,
        const double* input1,
        const double* input2, 
        double* output,
        size_t size,
        const std::vector<double>& params = {}
    );

    /**
     * @brief Register a new kernel
     * @param kernel_params Kernel definition
     */
    static void registerKernel(const KernelParams& kernel_params);

    /**
     * @brief Initialize all built-in kernels
     */
    static void initializeBuiltinKernels();

    /**
     * @brief Clean up resources
     */
    static void cleanup();

private:
#ifdef WITH_METAL
    struct MetalKernel {
        id<MTLComputePipelineState> pipeline;
        std::string source;
        size_t param_count;
    };
    
    static std::unordered_map<std::string, MetalKernel> metal_kernels_;
    static id<MTLDevice> device_;
    static id<MTLCommandQueue> command_queue_;
    
    static void compileMetalKernel(const std::string& name, const std::string& source, size_t param_count);
    static void executeMetalKernel(const std::string& name, const std::vector<id<MTLBuffer>>& buffers, size_t size);
#endif

    static bool initialized_;
    
    /**
     * @brief Generic buffer management
     */
    static std::vector<float> convertToFloat(const double* data, size_t size);
    static void convertFromFloat(const float* data, double* output, size_t size);
};

/**
 * @brief Activation function registry with automatic GPU kernel generation
 */
class ActivationKernelRegistry {
public:
    /**
     * @brief Activation function definition
     */
    struct ActivationDef {
        std::string name;
        std::string gpu_expression;  // e.g., "max(0.0f, input)" for ReLU
        std::vector<std::string> param_names;  // e.g., {"alpha"} for LeakyReLU
        bool has_parameters;
    };

    /**
     * @brief Register an activation function
     * @param def Activation function definition
     */
    static void registerActivation(const ActivationDef& def);

    /**
     * @brief Execute an activation function on GPU
     * @param name Activation function name
     * @param input Input data
     * @param output Output data
     * @param size Number of elements
     * @param params Parameters (if any)
     */
    static void executeActivation(
        const std::string& name,
        const double* input,
        double* output, 
        size_t size,
        const std::vector<double>& params = {}
    );

    /**
     * @brief Initialize all built-in activation functions
     */
    static void initializeBuiltinActivations();

private:
    static std::unordered_map<std::string, ActivationDef> activations_;
    
    /**
     * @brief Generate Metal kernel source from expression
     * @param name Function name
     * @param expression GPU expression
     * @param param_names Parameter names
     * @return Generated Metal kernel source
     */
    static std::string generateKernelSource(
        const std::string& name,
        const std::string& expression, 
        const std::vector<std::string>& param_names
    );
};

} // namespace Backend  
} // namespace MLLib
