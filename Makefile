# MLLib - C++ Machine Learning Library
# Makefile for building, testing, and formatting

# Project configuration
PROJECT_NAME = MLLib
VERSION = 1.0.0

# Directories
SRC_DIR = src
INCLUDE_DIR = include
BUILD_DIR = build
TEST_DIR = tests
SAMPLE_DIR = samples

# GPU vendor detection and flags
ROCM_AVAILABLE := $(shell which rocm-smi 2>/dev/null && echo true || echo false)
ONEAPI_AVAILABLE := $(shell which sycl-ls 2>/dev/null && echo true || echo false)
METAL_AVAILABLE := $(shell [ "$(shell uname)" = "Darwin" ] && echo true || echo false)

# Include CUDA configuration
include cuda.mk

# Compiler settings
CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2
DEBUG_FLAGS = -g -DDEBUG
INCLUDE_FLAGS = -I$(INCLUDE_DIR)

# CI environment: use clang++ for Objective-C++ support
ifdef CI
    CXX = clang++
endif

ifneq ($(GITHUB_ACTIONS),)
    CXX = clang++
endif

ifneq ($(GITLAB_CI),)
    CXX = clang++
endif

# GPU backend configuration
# CPU-only mode: Build without any GPU backend support
ifdef CPU_ONLY
    # Skip all GPU backend flags
    GPU_DISABLED = true
else
    # Enable all GPU support by default for library usage
    # Users can control GPU usage at runtime via device detection
    CXXFLAGS += -DWITH_CUDA -DWITH_ROCM -DWITH_ONEAPI -DWITH_METAL
endif

# Optional: Disable specific GPU backends at compile time
ifdef DISABLE_CUDA
    CXXFLAGS := $(filter-out -DWITH_CUDA,$(CXXFLAGS))
endif

ifdef DISABLE_ROCM
    CXXFLAGS := $(filter-out -DWITH_ROCM,$(CXXFLAGS))
endif

ifdef DISABLE_ONEAPI
    CXXFLAGS := $(filter-out -DWITH_ONEAPI,$(CXXFLAGS))
endif

ifdef DISABLE_METAL
    CXXFLAGS := $(filter-out -DWITH_METAL,$(CXXFLAGS))
endif

# Alternative: Disable all GPU backends at once
ifdef DISABLE_ALL_GPU
    CXXFLAGS := $(filter-out -DWITH_CUDA -DWITH_ROCM -DWITH_ONEAPI -DWITH_METAL,$(CXXFLAGS))
    GPU_DISABLED = true
endif

# Add CUDA flags if available
ifeq ($(CUDA_AVAILABLE),true)
    INCLUDE_FLAGS += $(CUDA_INCLUDE)
    LDFLAGS += $(CUDA_LIBS)
else
    # Add mock CUDA includes for stub implementation
    INCLUDE_FLAGS += -I/usr/local/cuda/include
endif

# Add ROCm flags if available
ifeq ($(ROCM_AVAILABLE),true)
    INCLUDE_FLAGS += -I/opt/rocm/include
    LDFLAGS += -L/opt/rocm/lib -lhipblas -lhip
else
    # Add mock ROCm includes for stub implementation
    INCLUDE_FLAGS += -I/opt/rocm/include
endif

# Add oneAPI flags if available
ifeq ($(ONEAPI_AVAILABLE),true)
    ifndef ONEAPI_ROOT
        ONEAPI_ROOT = /opt/intel/oneapi
    endif
    INCLUDE_FLAGS += -I$(ONEAPI_ROOT)/include
    LDFLAGS += -L$(ONEAPI_ROOT)/lib -lmkl_sycl -lmkl_intel_lp64 -lmkl_sequential -lmkl_core
else
    # Add mock oneAPI includes for stub implementation
    ifndef ONEAPI_ROOT
        ONEAPI_ROOT = /opt/intel/oneapi
    endif
    INCLUDE_FLAGS += -I$(ONEAPI_ROOT)/include
endif

# Add Metal flags (always on macOS, mock on other platforms)
ifeq ($(METAL_AVAILABLE),true)
    LDFLAGS += -framework Metal -framework Foundation -framework MetalPerformanceShaders
endif

# CI environment detection - disable problematic features
ifdef CI
    METAL_AVAILABLE = false
    # Remove Metal support flag in CI environment
    CXXFLAGS := $(filter-out -DWITH_METAL,$(CXXFLAGS))
    CPPFLAGS += -DCI
    $(info CI mode detected - disabling Metal support, using CPU/other GPU backends only)
endif

ifneq ($(GITHUB_ACTIONS),)
    METAL_AVAILABLE = false
    # Remove Metal support flag in GitHub Actions environment
    CXXFLAGS := $(filter-out -DWITH_METAL,$(CXXFLAGS))
    CPPFLAGS += -DCI
    $(info GitHub Actions detected - disabling Metal support, using CPU/other GPU backends only)
endif

ifneq ($(GITLAB_CI),)
    METAL_AVAILABLE = false
    # Remove Metal support flag in GitLab CI environment
    CXXFLAGS := $(filter-out -DWITH_METAL,$(CXXFLAGS))
    CPPFLAGS += -DCI
    $(info GitLab CI detected - disabling Metal support, using CPU/other GPU backends only)
endif

# Find source files
CPP_FILES = $(shell find $(SRC_DIR) -name "*.cpp" 2>/dev/null || true)
MM_FILES = $(shell find $(SRC_DIR) -name "*.mm" 2>/dev/null || true)
HPP_FILES = $(shell find $(INCLUDE_DIR) -name "*.hpp" 2>/dev/null || true)
SRC_HPP_FILES = $(shell find $(SRC_DIR) -name "*.hpp" 2>/dev/null || true)
TEST_FILES = $(shell find $(TEST_DIR) -name "*.cpp" 2>/dev/null || true)
TEST_HPP_FILES = $(shell find $(TEST_DIR) -name "*.hpp" 2>/dev/null || true)

# Override MM_FILES in CI environment after initial search
ifdef CI
    MM_FILES :=
endif
ifneq ($(GITHUB_ACTIONS),)
    MM_FILES :=
endif
ifneq ($(GITLAB_CI),)
    MM_FILES :=
endif

# Filter GPU backend files based on WITH_* flags (not availability)
# Include backend files if WITH_* is defined to ensure stub implementations are compiled
ifneq ($(findstring -DWITH_ROCM,$(CXXFLAGS)),)
    CPP_FILES += src/MLLib/backend/gpu/rocm_backend.cpp
endif

ifneq ($(findstring -DWITH_ONEAPI,$(CXXFLAGS)),)
    CPP_FILES += src/MLLib/backend/gpu/oneapi_backend.cpp
endif

# Only include Metal backend when Metal support is enabled and available
ifeq ($(METAL_AVAILABLE),true)
    ifneq ($(findstring -DWITH_METAL,$(CXXFLAGS)),)
        MM_FILES += src/MLLib/backend/gpu/metal_backend.mm
    endif
endif

# Add CUDA stub implementation when CUDA is not available but WITH_CUDA is enabled
ifeq ($(CUDA_AVAILABLE),false)
    ifneq ($(findstring -DWITH_CUDA,$(CXXFLAGS)),)
        CPP_FILES += src/MLLib/backend/gpu/cuda_kernels_stub.cpp
    endif
endif

# Remove duplicates from CPP_FILES to prevent linker errors
CPP_FILES := $(sort $(CPP_FILES))
MM_FILES := $(sort $(MM_FILES))

# Combine all source files
ALL_SOURCE_FILES = $(CPP_FILES) $(MM_FILES)

# Library target
LIB_NAME = lib$(PROJECT_NAME).a
LIB_TARGET = $(BUILD_DIR)/$(LIB_NAME)

# Tools for formatting and linting - support multiple installation paths
# Try different common paths for clang-format in order of preference
CLANG_FORMAT = $(shell \
	if command -v clang-format-20 >/dev/null 2>&1; then \
		echo "clang-format-20"; \
	elif command -v /usr/local/bin/clang-format >/dev/null 2>&1; then \
		echo "/usr/local/bin/clang-format"; \
	elif command -v clang-format >/dev/null 2>&1; then \
		echo "clang-format"; \
	else \
		echo "clang-format"; \
	fi)
CLANG_TIDY = clang-tidy
CPPCHECK = cppcheck

# Phony targets for main build, clean, test, and samples
.PHONY: all clean debug test gpu-integration-test samples run-sample xor device-detection gpu-vendor-detection gpu-test help install build-tools gpu-check

# Default target
.PHONY: all
all: $(LIB_TARGET)

# Create build directory
$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

# Build static library
$(LIB_TARGET): $(BUILD_DIR)
	@echo "Building $(PROJECT_NAME) library..."
	@echo $(CUDA_MESSAGE)
	@echo "C++ source files found: $(words $(CPP_FILES)) files"
	@echo "Objective-C++ source files found: $(words $(MM_FILES)) files"
	@echo "CUDA files found: $(words $(CUDA_FILES)) files"
	@echo "Header files found: $(words $(HPP_FILES)) files"
	@echo "Compiling source files..."
	@echo "Compiler: $(CXX)"
	@echo "Flags: $(CXXFLAGS) $(INCLUDE_FLAGS)"
	@if [ -n "$(CPP_FILES)" ]; then \
		echo "Compiling C++ files..."; \
		$(CXX) $(CXXFLAGS) $(INCLUDE_FLAGS) -c $(CPP_FILES); \
	fi
	@if [ -n "$(MM_FILES)" ] && [ "$(METAL_AVAILABLE)" = "true" ]; then \
		echo "Compiling Objective-C++ files..."; \
		$(CXX) $(CXXFLAGS) $(INCLUDE_FLAGS) -c $(MM_FILES); \
	elif [ -n "$(MM_FILES)" ] && [ -n "$(CI)" ]; then \
		echo "Checking CI Objective-C++ compiler availability..."; \
		if $(CXX) --help | grep -q objc 2>/dev/null; then \
			echo "Compiling Objective-C++ files (CI mode with native implementation)..."; \
			$(CXX) $(CXXFLAGS) $(INCLUDE_FLAGS) -c $(MM_FILES); \
		else \
			echo "Skipping Objective-C++ files (compiler not available in CI)"; \
			echo "Metal backend stub will be used instead"; \
		fi; \
	elif [ -n "$(MM_FILES)" ]; then \
		echo "Skipping Objective-C++ files (Metal not available)"; \
	fi
	@if [ "$(CUDA_AVAILABLE)" = "true" ] && [ -n "$(CUDA_FILES)" ]; then \
		echo "Compiling CUDA files..."; \
		for cu_file in $(CUDA_FILES); do \
			obj_file=$$(basename $$cu_file .cu).o; \
			echo "  nvcc $$cu_file -> $$obj_file"; \
			$(NVCC) $(NVCC_FLAGS) $(INCLUDE_FLAGS) $(CUDA_INCLUDE) -c $$cu_file -o $$obj_file; \
		done; \
	fi
	@echo "Creating static library..."
	@ar rcs $(LIB_TARGET) *.o
	@rm -f *.o
	@echo "✅ Library built successfully: $(LIB_TARGET)"
	@ls -la $(LIB_TARGET)

# Build with debug flags
.PHONY: debug
debug: CXXFLAGS += $(DEBUG_FLAGS)
debug: $(LIB_TARGET)

# Clean build artifacts
.PHONY: clean
clean:
	@echo "Cleaning build artifacts..."
	@if [ -d "$(BUILD_DIR)" ]; then \
		echo "Removing build directory: $(BUILD_DIR)"; \
		rm -rf $(BUILD_DIR); \
	else \
		echo "Build directory $(BUILD_DIR) does not exist"; \
	fi
	@rm -f *.o *.a *.so *.dylib
	@find . -name "*.o" -delete 2>/dev/null || true
	@find . -name "core" -delete 2>/dev/null || true
	@echo "Removing training output directories..."
	@for dir in samples/training_*; do \
		if [ -d "$$dir" ]; then \
			echo "Removing training directory: $$dir"; \
			rm -rf "$$dir"; \
		fi; \
	done
	@echo "Removing test temporary files..."
	@find . -name "mllib_test_*" -delete 2>/dev/null || true
	@echo "✅ Clean completed"

# Deep clean - remove all generated files
.PHONY: deep-clean
deep-clean: clean
	@echo "Performing deep clean..."
	@rm -rf .vscode/settings.json.backup
	@find . -name "*.tmp" -delete 2>/dev/null || true
	@find . -name "*.log" -delete 2>/dev/null || true
	@find . -name ".DS_Store" -delete 2>/dev/null || true
	@echo "Removing any remaining training directories..."
	@find . -type d -name "training_*" -exec rm -rf {} + 2>/dev/null || true
	@echo "✅ Deep clean completed"

# Format code using clang-format
.PHONY: fmt
fmt:
	@echo "Formatting C++ code with clang-format..."
	@if command -v $(CLANG_FORMAT) >/dev/null 2>&1; then \
		if [ -n "$(HPP_FILES)" ]; then \
			$(CLANG_FORMAT) -i $(HPP_FILES); \
			echo "✅ Include header files formatted"; \
		fi; \
		if [ -n "$(SRC_HPP_FILES)" ]; then \
			$(CLANG_FORMAT) -i $(SRC_HPP_FILES); \
			echo "✅ Source header files formatted"; \
		fi; \
		if [ -n "$(CPP_FILES)" ]; then \
			$(CLANG_FORMAT) -i $(CPP_FILES); \
			echo "✅ Source files formatted"; \
		fi; \
		if [ -n "$(TEST_FILES)" ]; then \
			$(CLANG_FORMAT) -i $(TEST_FILES); \
			echo "✅ Test source files formatted"; \
		fi; \
		if [ -n "$(TEST_HPP_FILES)" ]; then \
			$(CLANG_FORMAT) -i $(TEST_HPP_FILES); \
			echo "✅ Test header files formatted"; \
		fi; \
		echo "✅ Code formatting completed"; \
	else \
		echo "❌ clang-format not found. Install with: brew install clang-format"; \
	fi

# Check format without modifying files
.PHONY: fmt-check
fmt-check:
	@echo "Checking code format..."
	@echo "Detected CLANG_FORMAT: $(CLANG_FORMAT)"
	@if command -v $(CLANG_FORMAT) >/dev/null 2>&1; then \
		echo "✅ clang-format found: $(CLANG_FORMAT)"; \
		$(CLANG_FORMAT) --version; \
		FORMAT_NEEDED=0; \
		if [ -n "$(HPP_FILES)" ]; then \
			for file in $(HPP_FILES); do \
				if ! $(CLANG_FORMAT) --dry-run --Werror $$file >/dev/null 2>&1; then \
					echo "❌ $$file needs formatting"; \
					FORMAT_NEEDED=1; \
				fi; \
			done; \
		fi; \
		if [ -n "$(SRC_HPP_FILES)" ]; then \
			for file in $(SRC_HPP_FILES); do \
				if ! $(CLANG_FORMAT) --dry-run --Werror $$file >/dev/null 2>&1; then \
					echo "❌ $$file needs formatting"; \
					FORMAT_NEEDED=1; \
				fi; \
			done; \
		fi; \
		if [ -n "$(CPP_FILES)" ]; then \
			for file in $(CPP_FILES); do \
				if ! $(CLANG_FORMAT) --dry-run --Werror $$file >/dev/null 2>&1; then \
					echo "❌ $$file needs formatting"; \
					FORMAT_NEEDED=1; \
				fi; \
			done; \
		fi; \
		if [ -n "$(TEST_FILES)" ]; then \
			for file in $(TEST_FILES); do \
				if ! $(CLANG_FORMAT) --dry-run --Werror $$file >/dev/null 2>&1; then \
					echo "❌ $$file needs formatting"; \
					FORMAT_NEEDED=1; \
				fi; \
			done; \
		fi; \
		if [ -n "$(TEST_HPP_FILES)" ]; then \
			for file in $(TEST_HPP_FILES); do \
				if ! $(CLANG_FORMAT) --dry-run --Werror $$file >/dev/null 2>&1; then \
					echo "❌ $$file needs formatting"; \
					FORMAT_NEEDED=1; \
				fi; \
			done; \
		fi; \
		if [ $$FORMAT_NEEDED -eq 0 ]; then \
			echo "✅ All files are properly formatted"; \
		else \
			echo "❌ Some files need formatting. Run 'make fmt'"; \
			exit 1; \
		fi; \
	else \
		echo "❌ clang-format not found"; \
		echo "Diagnostic information:"; \
		echo "  CLANG_FORMAT variable: $(CLANG_FORMAT)"; \
		echo "  Searching in common locations:"; \
		ls -la /usr/bin/clang-format* 2>/dev/null || echo "  - Not found in /usr/bin/"; \
		ls -la /usr/local/bin/clang-format* 2>/dev/null || echo "  - Not found in /usr/local/bin/"; \
		which clang-format 2>/dev/null || echo "  - Not found in PATH"; \
		echo "  Available packages:"; \
		apt list --installed | grep clang 2>/dev/null || echo "  - No clang packages found"; \
		exit 1; \
	fi

# Lint code using clang-tidy
.PHONY: lint
lint:
	@echo "Running lint checks with clang-tidy..."
	@if command -v $(CLANG_TIDY) >/dev/null 2>&1; then \
		if [ -n "$(CPP_FILES)" ]; then \
			$(CLANG_TIDY) $(CPP_FILES) -- $(CXXFLAGS) $(INCLUDE_FLAGS); \
		fi; \
		if [ -n "$(SRC_HPP_FILES)" ]; then \
			for file in $(SRC_HPP_FILES); do \
				echo "Checking $$file..."; \
				$(CLANG_TIDY) $$file -- $(CXXFLAGS) $(INCLUDE_FLAGS); \
			done; \
		fi; \
		if [ -n "$(HPP_FILES)" ]; then \
			for file in $(HPP_FILES); do \
				echo "Checking $$file..."; \
				if [[ "$$file" == *"metal_backend.hpp" ]]; then \
					echo "Skipping Metal backend (requires Objective-C++ context)"; \
				else \
					$(CLANG_TIDY) $$file -- $(CXXFLAGS) $(INCLUDE_FLAGS); \
				fi; \
			done; \
		fi; \
		if [ -n "$(TEST_HPP_FILES)" ]; then \
			for file in $(TEST_HPP_FILES); do \
				echo "Checking $$file..."; \
				$(CLANG_TIDY) $$file -- $(CXXFLAGS) $(INCLUDE_FLAGS); \
			done; \
		fi; \
		echo "✅ Lint checks completed"; \
	else \
		echo "❌ clang-tidy not found. Install with: brew install llvm"; \
	fi

# Static analysis with cppcheck
.PHONY: check
check:
	@echo "Running static analysis with cppcheck..."
	@if command -v $(CPPCHECK) >/dev/null 2>&1; then \
		$(CPPCHECK) --enable=all --std=c++17 \
			-I$(INCLUDE_DIR) \
			--suppress=missingIncludeSystem \
			--suppress=unusedFunction \
			$(SRC_DIR) $(INCLUDE_DIR) 2>/dev/null; \
		echo "✅ Static analysis completed"; \
	else \
		echo "❌ cppcheck not found. Install with: brew install cppcheck"; \
	fi

# Combined linting (format + lint + check)
.PHONY: lint-all
lint-all: fmt lint check
	@echo "✅ All linting and checks completed"

# Install formatting tools
.PHONY: install-tools
install-tools:
	@echo "Installing code formatting and linting tools..."
	@if command -v brew >/dev/null 2>&1; then \
		brew install clang-format llvm cppcheck; \
		echo "✅ Tools installed successfully"; \
	else \
		echo "❌ Homebrew not found. Please install tools manually:"; \
		echo "  - clang-format: for code formatting"; \
		echo "  - clang-tidy: for linting (part of LLVM)"; \
		echo "  - cppcheck: for static analysis"; \
	fi

# Run all tests (unit + integration)
.PHONY: test
test: unit-test integration-test
	@echo "✅ All tests completed successfully"

# Run GPU integration tests
.PHONY: gpu-integration-test
gpu-integration-test: $(LIB_TARGET)
	@echo "Building and running GPU integration tests..."
	@mkdir -p $(BUILD_DIR)/tests
	@if [ -f "$(TEST_DIR)/integration/gpu_integration_test.cpp" ]; then \
		echo "Compiling GPU integration test..."; \
		GPU_TEST_FILES="$(TEST_DIR)/common/test_utils.cpp $(TEST_DIR)/integration/gpu_integration_test.cpp"; \
		COMPILE_CMD="$(CXX) $(CXXFLAGS) $(INCLUDE_FLAGS) $$GPU_TEST_FILES -L$(BUILD_DIR) -lMLLib"; \
		if [ "$(shell uname)" = "Darwin" ]; then \
			echo "Building with Metal support for GPU integration tests..."; \
			COMPILE_CMD="$$COMPILE_CMD -framework Metal -framework Foundation -framework MetalPerformanceShaders"; \
		fi; \
		if [ "$(CUDA_AVAILABLE)" = "true" ]; then \
			echo "Building with CUDA support for tests..."; \
			COMPILE_CMD="$$COMPILE_CMD $(LDFLAGS)"; \
		else \
			echo "Building without CUDA support for tests..."; \
			if [ "$(ROCM_AVAILABLE)" = "true" ]; then \
				COMPILE_CMD="$$COMPILE_CMD -L/opt/rocm/lib -lhipblas -lhip"; \
			fi; \
			if [ "$(ONEAPI_AVAILABLE)" = "true" ]; then \
				COMPILE_CMD="$$COMPILE_CMD -L$(ONEAPI_ROOT)/lib -lmkl_sycl -lmkl_intel_lp64 -lmkl_sequential -lmkl_core"; \
			fi; \
		fi; \
		if $$COMPILE_CMD -o $(BUILD_DIR)/tests/gpu_integration_test; then \
			echo "Running GPU integration tests..."; \
			if $(BUILD_DIR)/tests/gpu_integration_test; then \
				echo "✅ GPU integration tests passed"; \
			else \
				echo "❌ GPU integration tests failed"; \
				exit 1; \
			fi; \
		else \
			echo "❌ Failed to build GPU integration tests"; \
			exit 1; \
		fi; \
	else \
		echo "ℹ️  GPU integration tests not found"; \
	fi

# Run unit tests only
.PHONY: unit-test
unit-test: $(LIB_TARGET)
	@echo "Building and running unit tests..."
	@mkdir -p $(BUILD_DIR)/tests
	@if [ -f "$(TEST_DIR)/unit/unit_test_main.cpp" ]; then \
		echo "Compiling unit test framework..."; \
		UNIT_FILES="$(TEST_DIR)/common/test_utils.cpp $(TEST_DIR)/unit/unit_test_main.cpp"; \
		COMPILE_CMD="$(CXX) $(CXXFLAGS) $(INCLUDE_FLAGS) $$UNIT_FILES -L$(BUILD_DIR) -lMLLib"; \
		if [ "$(shell uname)" = "Darwin" ]; then \
			echo "Building with Metal support for unit tests..."; \
			COMPILE_CMD="$$COMPILE_CMD -framework Metal -framework Foundation -framework MetalPerformanceShaders"; \
		fi; \
		if [ "$(CUDA_AVAILABLE)" = "true" ]; then \
			echo "Building with CUDA support for tests..."; \
			COMPILE_CMD="$$COMPILE_CMD $(LDFLAGS)"; \
		else \
			echo "Building without CUDA support for tests..."; \
			if [ "$(ROCM_AVAILABLE)" = "true" ]; then \
				COMPILE_CMD="$$COMPILE_CMD -L/opt/rocm/lib -lhipblas -lhip"; \
			fi; \
			if [ "$(ONEAPI_AVAILABLE)" = "true" ]; then \
				COMPILE_CMD="$$COMPILE_CMD -L$(ONEAPI_ROOT)/lib -lmkl_sycl -lmkl_intel_lp64 -lmkl_sequential -lmkl_core"; \
			fi; \
		fi; \
		if $$COMPILE_CMD -o $(BUILD_DIR)/tests/unit_tests; then \
			echo "Running unit tests..."; \
			if $(BUILD_DIR)/tests/unit_tests; then \
				echo "✅ Unit tests passed"; \
			else \
				echo "❌ Unit tests failed"; \
				exit 1; \
			fi; \
		else \
			echo "❌ Failed to build unit tests"; \
			exit 1; \
		fi; \
	else \
		echo "ℹ️  Unit tests not found"; \
	fi

# Run simple integration tests
.PHONY: simple-integration-test
simple-integration-test: $(LIB_TARGET)
	@echo "Building and running simple integration tests..."
	@echo "Compiling simple integration test..."
	@SIMPLE_COMPILE_CMD="$(CXX) $(CXXFLAGS) -I$(INCLUDE_DIR) -I$(TEST_DIR) -o $(BUILD_DIR)/simple_integration_test tests/integration/simple_integration_test.cpp -L$(BUILD_DIR) -lMLLib"; \
	if [ "$(shell uname)" = "Darwin" ]; then \
		echo "Building with Metal support for simple integration tests..."; \
		SIMPLE_COMPILE_CMD="$$SIMPLE_COMPILE_CMD -framework Metal -framework Foundation -framework MetalPerformanceShaders"; \
	fi; \
	if [ "$(CUDA_AVAILABLE)" = "true" ]; then \
		echo "Building with CUDA support for tests..."; \
		SIMPLE_COMPILE_CMD="$$SIMPLE_COMPILE_CMD $(LDFLAGS)"; \
	else \
		echo "Building without CUDA support for tests..."; \
		if [ "$(ROCM_AVAILABLE)" = "true" ]; then \
			SIMPLE_COMPILE_CMD="$$SIMPLE_COMPILE_CMD -L/opt/rocm/lib -lhipblas -lhip"; \
		fi; \
		if [ "$(ONEAPI_AVAILABLE)" = "true" ]; then \
			SIMPLE_COMPILE_CMD="$$SIMPLE_COMPILE_CMD -L$(ONEAPI_ROOT)/lib -lmkl_sycl -lmkl_intel_lp64 -lmkl_sequential -lmkl_core"; \
		fi; \
	fi; \
	$$SIMPLE_COMPILE_CMD 2>&1 | tee $(BUILD_DIR)/simple_integration_compile.log; \
	if [ $$? -eq 0 ]; then \
		echo "✅ Simple integration tests compiled successfully"; \
		echo "Running simple integration tests..."; \
		$(BUILD_DIR)/simple_integration_test; \
		if [ $$? -eq 0 ]; then \
			echo "✅ Simple integration tests completed successfully"; \
		else \
			echo "❌ Simple integration tests failed"; \
			exit 1; \
		fi; \
	else \
		echo "❌ Failed to build simple integration tests"; \
		echo "Compilation errors:"; \
		cat $(BUILD_DIR)/simple_integration_compile.log; \
		exit 1; \
	fi

# Run integration tests only
.PHONY: integration-test
integration-test: $(LIB_TARGET)
	@echo "Building and running integration tests..."
	@mkdir -p $(BUILD_DIR)/tests
	@if [ -f "$(TEST_DIR)/integration/integration_test_main.cpp" ]; then \
		echo "Compiling integration test framework..."; \
		INTEGRATION_FILES="$(TEST_DIR)/common/test_utils.cpp $(TEST_DIR)/integration/integration_test_main.cpp"; \
		COMPILE_CMD="$(CXX) $(CXXFLAGS) $(INCLUDE_FLAGS) $$INTEGRATION_FILES $(LIB_TARGET)"; \
		if [ "$(shell uname)" = "Darwin" ]; then \
			echo "Building with Metal support for integration tests..."; \
			COMPILE_CMD="$$COMPILE_CMD -framework Metal -framework Foundation -framework MetalPerformanceShaders"; \
		fi; \
		if [ "$(CUDA_AVAILABLE)" = "true" ]; then \
			echo "Building with CUDA support for tests..."; \
			COMPILE_CMD="$$COMPILE_CMD $(LDFLAGS)"; \
		else \
			echo "Building without CUDA support for tests..."; \
			if [ "$(ROCM_AVAILABLE)" = "true" ]; then \
				COMPILE_CMD="$$COMPILE_CMD -L/opt/rocm/lib -lhipblas -lhip"; \
			fi; \
			if [ "$(ONEAPI_AVAILABLE)" = "true" ]; then \
				COMPILE_CMD="$$COMPILE_CMD -L$(ONEAPI_ROOT)/lib -lmkl_sycl -lmkl_intel_lp64 -lmkl_sequential -lmkl_core"; \
			fi; \
		fi; \
		if $$COMPILE_CMD -o $(BUILD_DIR)/tests/integration_tests; then \
			echo "Running integration tests..."; \
			if $(BUILD_DIR)/tests/integration_tests; then \
				echo "✅ Integration tests passed"; \
			else \
				echo "❌ Integration tests failed"; \
				exit 1; \
			fi; \
		else \
			echo "❌ Failed to build integration tests"; \
			exit 1; \
		fi; \
	else \
		echo "ℹ️  Integration tests not found"; \
	fi

# Run comprehensive test runner
.PHONY: test-all
test-all: $(LIB_TARGET)
	@echo "Building comprehensive test runner..."
	@mkdir -p $(BUILD_DIR)/tests
	@if [ -f "$(TEST_DIR)/test_main.cpp" ]; then \
		echo "Compiling test runner..."; \
		if $(CXX) $(CXXFLAGS) $(INCLUDE_FLAGS) $(TEST_DIR)/test_main.cpp -o $(BUILD_DIR)/tests/test_runner; then \
			echo "Running comprehensive test suite..."; \
			if $(BUILD_DIR)/tests/test_runner; then \
				echo "✅ All tests completed successfully"; \
			else \
				echo "❌ Some tests failed"; \
				exit 1; \
			fi; \
		else \
			echo "❌ Failed to build test runner"; \
			exit 1; \
		fi; \
	else \
		echo "ℹ️  Test runner not found"; \
	fi

# Build samples (if sample directory exists)
.PHONY: samples
samples: $(LIB_TARGET)
	@if [ -d "$(SAMPLE_DIR)" ]; then \
		echo "Building samples..."; \
		mkdir -p $(BUILD_DIR)/samples; \
		SAMPLE_SUCCESS=true; \
		for sample in $(SAMPLE_DIR)/*.cpp; do \
			if [ -f "$$sample" ]; then \
				name=$$(basename $$sample .cpp); \
				\
				if [ "$(CI)" = "1" ] && [ "$$name" = "gpu_usage_test" ]; then \
					echo "Skipping sample: $$name (requires Metal in CI)"; \
					continue; \
				fi; \
				\
				echo "Building sample: $$name"; \
				SAMPLE_COMPILE_CMD="$(CXX) $(CXXFLAGS) $(INCLUDE_FLAGS) $$sample $(LIB_TARGET)"; \
				if [ "$(shell uname)" = "Darwin" ] && [ "$(METAL_AVAILABLE)" = "true" ]; then \
					echo "Building with Metal support for sample: $$name"; \
					SAMPLE_COMPILE_CMD="$$SAMPLE_COMPILE_CMD -framework Metal -framework Foundation -framework MetalPerformanceShaders"; \
				fi; \
				if [ "$(CUDA_AVAILABLE)" = "true" ]; then \
					echo "Building with CUDA support for sample: $$name"; \
					SAMPLE_COMPILE_CMD="$$SAMPLE_COMPILE_CMD $(LDFLAGS)"; \
				else \
					echo "Building without CUDA support for sample: $$name"; \
					if [ "$(ROCM_AVAILABLE)" = "true" ]; then \
						SAMPLE_COMPILE_CMD="$$SAMPLE_COMPILE_CMD -L/opt/rocm/lib -lhipblas -lhip"; \
					fi; \
					if [ "$(ONEAPI_AVAILABLE)" = "true" ]; then \
						SAMPLE_COMPILE_CMD="$$SAMPLE_COMPILE_CMD -L$(ONEAPI_ROOT)/lib -lmkl_sycl -lmkl_intel_lp64 -lmkl_sequential -lmkl_core"; \
					fi; \
				fi; \
				if $$SAMPLE_COMPILE_CMD -o $(BUILD_DIR)/samples/$$name; then \
					echo "✓ Built sample: $$name"; \
				else \
					echo "❌ Failed to build sample: $$name"; \
					SAMPLE_SUCCESS=false; \
				fi; \
			fi; \
		done; \
		if [ "$$SAMPLE_SUCCESS" = "true" ]; then \
			echo "✅ Samples built successfully"; \
		else \
			echo "❌ Some samples failed to build"; \
			exit 1; \
		fi; \
	else \
		echo "ℹ️  No samples directory found"; \
	fi

# Generic sample runner
.PHONY: run-sample
run-sample:
	@if [ -z "$(SAMPLE)" ]; then \
		echo "Usage: make run-sample SAMPLE=<sample_name>"; \
		echo "Available samples:"; \
		if [ -d "$(BUILD_DIR)/samples" ]; then \
			for sample in $(BUILD_DIR)/samples/*; do \
				if [ -f "$$sample" ] && [ -x "$$sample" ]; then \
					echo "  $$(basename $$sample)"; \
				fi; \
			done; \
		else \
			echo "  (no samples built yet - run 'make samples' first)"; \
		fi; \
	elif [ -f "$(BUILD_DIR)/samples/$(SAMPLE)" ]; then \
		echo "Running $(SAMPLE) sample..."; \
		$(BUILD_DIR)/samples/$(SAMPLE); \
	else \
		echo "❌ Sample '$(SAMPLE)' not found"; \
		echo "Available samples:"; \
		if [ -d "$(BUILD_DIR)/samples" ]; then \
			for sample in $(BUILD_DIR)/samples/*; do \
				if [ -f "$$sample" ] && [ -x "$$sample" ]; then \
					echo "  $$(basename $$sample)"; \
				fi; \
			done; \
		fi; \
		echo "Build samples first with 'make samples'"; \
	fi

# Convenience aliases for commonly used samples
.PHONY: xor device-detection gpu-vendor-detection gpu-test
xor: samples
	@$(MAKE) run-sample SAMPLE=xor

gpu-test: gpu-integration-test

.PHONY: device-detection
device-detection: samples
	@$(MAKE) run-sample SAMPLE=device_detection

.PHONY: gpu-vendor-detection
gpu-vendor-detection: samples
	@$(MAKE) run-sample SAMPLE=gpu_vendor_detection

# CI-specific build target - builds CPU-only version with all stubs
.PHONY: ci-build
ci-build: 
	@$(MAKE) CI=1 $(LIB_TARGET)
	@echo "✅ CI build completed successfully"

# CI-specific test target
.PHONY: ci-test
ci-test: 
	@$(MAKE) CI=1 unit-test simple-integration-test
	@echo "✅ CI tests completed successfully"
.PHONY: help
help:
	@echo "$(PROJECT_NAME) v$(VERSION) - Available targets:"
	@echo ""
	@echo "Building:"
	@echo "  all          - Build the library (default)"
	@echo "  debug        - Build with debug flags"
	@echo "  ci-build     - Build for CI environment (no Metal, GPU simulation)"
	@echo "  clean        - Remove build artifacts, training outputs, and test files"
	@echo "  deep-clean   - Remove all generated files including nested training dirs"
	@echo ""
	@echo "Testing:"
	@echo "  test         - Run all tests (unit + integration)"
	@echo "  unit-test    - Run unit tests only"
	@echo "  integration-test - Run integration tests only"
	@echo "  gpu-integration-test - Run GPU integration tests only"
	@echo "  test-all     - Run comprehensive test runner"
	@echo "  ci-test      - Run tests in CI environment"
	@echo ""
	@echo "Code Quality:"
	@echo "  fmt          - Format code with clang-format"
	@echo "  fmt-check    - Check if code is properly formatted"
	@echo "  lint         - Run clang-tidy linting"
	@echo "  check        - Run cppcheck static analysis"
	@echo "  lint-all     - Run all formatting and linting"
	@echo ""
	@echo "Tools:"
	@echo "  install-tools - Install required formatting/linting tools"
	@echo ""
	@echo "Samples:"
	@echo "  samples           - Build all sample programs"
	@echo "  run-sample SAMPLE=<name> - Run specific sample"
	@echo "  xor               - Build and run XOR sample (alias)"
	@echo "  device-detection  - Build and run device detection sample (alias)"
	@echo "  gpu-vendor-detection - Build and run GPU vendor detection sample (alias)"
	@echo "  gpu-test          - Run GPU integration tests (alias)"
	@echo ""
	@echo "GPU Configuration:"
	@echo "  Default: All GPU backends enabled (CUDA, ROCm, oneAPI, Metal)"
	@echo "  GPU detection and usage controlled at runtime"
	@echo ""
	@echo "  CPU_ONLY=1        - Build without any GPU backend support (CPU only)"
	@echo "  DISABLE_ALL_GPU=1 - Disable all GPU backends at compile time"
	@echo "  DISABLE_CUDA=1    - Disable CUDA support at compile time"
	@echo "  DISABLE_ROCM=1    - Disable AMD ROCm support at compile time" 
	@echo "  DISABLE_ONEAPI=1  - Disable Intel oneAPI support at compile time"
	@echo "  DISABLE_METAL=1   - Disable Apple Metal support at compile time"
	@echo ""
	@echo "Usage Examples:"
	@echo "  make                     - Build with all GPU support (runtime detection)"
	@echo "  make CPU_ONLY=1          - Build CPU-only version (no GPU backends)"
	@echo "  make DISABLE_ALL_GPU=1   - Build without GPU backends"
	@echo "  make DISABLE_CUDA=1      - Build without CUDA backend"
	@echo "  make ci-build            - Build for CI (CPU-only with stubs)"
	@echo "  make ci-test             - Run tests in CI environment"
	@echo "  make samples             - Build all sample programs"
	@echo "  make xor                 - Run XOR sample (auto GPU detection)"
	@echo "  make test                - Run tests with runtime GPU detection"
	@echo ""
	@echo "Misc:"
	@echo "  help         - Show this help message"