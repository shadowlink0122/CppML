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
EXAMPLE_DIR = examples
SAMPLE_DIR = samples

# Include CUDA configuration
include cuda.mk

# Compiler settings
CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2
DEBUG_FLAGS = -g -DDEBUG
INCLUDE_FLAGS = -I$(INCLUDE_DIR)

# Add CUDA flags if available
ifeq ($(CUDA_AVAILABLE),true)
    CXXFLAGS += -DWITH_CUDA
    INCLUDE_FLAGS += $(CUDA_INCLUDE)
    LDFLAGS += $(CUDA_LIBS)
endif

# Force CUDA support when WITH_CUDA=1 is specified, even without nvcc
ifdef WITH_CUDA
    ifeq ($(WITH_CUDA),1)
        CXXFLAGS += -DWITH_CUDA
        ifneq ($(CUDA_AVAILABLE),true)
            # Mock CUDA includes when nvcc not available but WITH_CUDA=1
            INCLUDE_FLAGS += -I/usr/local/cuda/include
        endif
    endif
endif

# Add GPU simulation flags if specified
ifdef GPU_SIMULATION
    CXXFLAGS += -DGPU_SIMULATION_MODE
    CUDA_AVAILABLE = true
    CXXFLAGS += -DWITH_CUDA
    INCLUDE_FLAGS += -I/usr/local/cuda/include
endif

# Find source files
CPP_FILES = $(shell find $(SRC_DIR) -name "*.cpp" 2>/dev/null || true)
HPP_FILES = $(shell find $(INCLUDE_DIR) -name "*.hpp" 2>/dev/null || true)
TEST_FILES = $(shell find $(TEST_DIR) -name "*.cpp" 2>/dev/null || true)

# Add CUDA stub implementation when CUDA is not available but WITH_CUDA is requested
ifeq ($(CUDA_AVAILABLE),false)
    # Only add stub once, regardless of multiple conditions
    NEED_CUDA_STUB := false
    ifneq ($(WITH_CUDA),)
        ifeq ($(WITH_CUDA),1)
            NEED_CUDA_STUB := true
        endif
    endif
    ifdef GPU_SIMULATION
        NEED_CUDA_STUB := true
    endif
    
    ifeq ($(NEED_CUDA_STUB),true)
        CPP_FILES += src/MLLib/backend/gpu/cuda_kernels_stub.cpp
    endif
endif

# Remove duplicates from CPP_FILES to prevent linker errors
CPP_FILES := $(sort $(CPP_FILES))

# Library target
LIB_NAME = lib$(PROJECT_NAME).a
LIB_TARGET = $(BUILD_DIR)/$(LIB_NAME)

# Tools for formatting and linting
CLANG_FORMAT = clang-format
CLANG_TIDY = clang-tidy
CPPCHECK = cppcheck

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
	@echo "Source files found: $(CPP_FILES)"
	@echo "CUDA files found: $(CUDA_FILES)"
	@echo "Header files found: $(HPP_FILES)"
	@if [ -n "$(CPP_FILES)" ] || [ -n "$(CUDA_FILES)" ]; then \
		echo "Compiling source files..."; \
		echo "Compiler: $(CXX)"; \
		echo "Flags: $(CXXFLAGS) $(INCLUDE_FLAGS)"; \
		if $(CXX) $(CXXFLAGS) $(INCLUDE_FLAGS) -c $(CPP_FILES) 2>&1; then \
			if [ "$(CUDA_AVAILABLE)" = "true" ]; then \
				echo "Compiling CUDA files..."; \
				for cu_file in $(CUDA_FILES); do \
					obj_file=$$(basename $$cu_file .cu).o; \
					echo "  nvcc $$cu_file -> $$obj_file"; \
					$(NVCC) $(NVCC_FLAGS) $(INCLUDE_FLAGS) $(CUDA_INCLUDE) -c $$cu_file -o $$obj_file; \
				done; \
			fi; \
			echo "Creating static library..."; \
			if ar rcs $(LIB_TARGET) *.o 2>&1; then \
				rm -f *.o; \
				echo "✅ Library built successfully: $(LIB_TARGET)"; \
				ls -la $(LIB_TARGET); \
			else \
				echo "❌ Failed to create static library"; \
				rm -f *.o; \
				exit 1; \
			fi; \
		else \
			echo "❌ Compilation failed"; \
			rm -f *.o; \
			exit 1; \
		fi; \
	else \
		echo "ℹ️  No source files found, creating header-only library marker"; \
		touch $(LIB_TARGET); \
		echo "✅ Header-only library marker created: $(LIB_TARGET)"; \
	fi

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
			echo "✅ Header files formatted"; \
		fi; \
		if [ -n "$(CPP_FILES)" ]; then \
			$(CLANG_FORMAT) -i $(CPP_FILES); \
			echo "✅ Source files formatted"; \
		fi; \
		if [ -n "$(TEST_FILES)" ]; then \
			$(CLANG_FORMAT) -i $(TEST_FILES); \
			echo "✅ Test files formatted"; \
		fi; \
		echo "✅ Code formatting completed"; \
	else \
		echo "❌ clang-format not found. Install with: brew install clang-format"; \
	fi

# Check format without modifying files
.PHONY: fmt-check
fmt-check:
	@echo "Checking code format..."
	@if command -v $(CLANG_FORMAT) >/dev/null 2>&1; then \
		FORMAT_NEEDED=0; \
		if [ -n "$(HPP_FILES)" ]; then \
			for file in $(HPP_FILES); do \
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
		if [ $$FORMAT_NEEDED -eq 0 ]; then \
			echo "✅ All files are properly formatted"; \
		else \
			echo "❌ Some files need formatting. Run 'make fmt'"; \
			exit 1; \
		fi; \
	else \
		echo "❌ clang-format not found"; \
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
		if [ -n "$(HPP_FILES)" ]; then \
			for file in $(HPP_FILES); do \
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
			--include-path=$(INCLUDE_DIR) \
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

# Run unit tests only
.PHONY: unit-test
unit-test: $(LIB_TARGET)
	@echo "Building and running unit tests..."
	@mkdir -p $(BUILD_DIR)/tests
	@if [ -f "$(TEST_DIR)/unit/unit_test_main.cpp" ]; then \
		echo "Compiling unit test framework..."; \
		UNIT_FILES="$(TEST_DIR)/common/test_utils.cpp $(TEST_DIR)/unit/unit_test_main.cpp"; \
		COMPILE_CMD="$(CXX) $(CXXFLAGS) $(INCLUDE_FLAGS) $$UNIT_FILES -L$(BUILD_DIR) -lMLLib"; \
		if [ "$(CUDA_AVAILABLE)" = "true" ] || [ "$(WITH_CUDA)" = "1" ] || [ -n "$(GPU_SIMULATION)" ]; then \
			echo "Building with CUDA support for tests..."; \
			COMPILE_CMD="$$COMPILE_CMD $(LDFLAGS)"; \
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
	@$(CXX) $(CXXFLAGS) -I$(INCLUDE_DIR) -I$(TEST_DIR) -o $(BUILD_DIR)/simple_integration_test \
		tests/integration/simple_integration_test.cpp -L$(BUILD_DIR) -lMLLib $(LDFLAGS) 2>&1 | tee $(BUILD_DIR)/simple_integration_compile.log; \
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
		COMPILE_CMD="$(CXX) $(CXXFLAGS) $(INCLUDE_FLAGS) $$INTEGRATION_FILES $(CPP_FILES)"; \
		if [ "$(CUDA_AVAILABLE)" = "true" ] || [ "$(WITH_CUDA)" = "1" ] || [ -n "$(GPU_SIMULATION)" ]; then \
			echo "Building with CUDA support for tests..."; \
			COMPILE_CMD="$$COMPILE_CMD $(LDFLAGS)"; \
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

# Build examples (if examples directory exists)
.PHONY: examples
examples: $(LIB_TARGET)
	@if [ -d "$(EXAMPLE_DIR)" ]; then \
		echo "Building examples..."; \
		mkdir -p $(BUILD_DIR)/examples; \
		EXAMPLE_SUCCESS=true; \
		for example in $(EXAMPLE_DIR)/*.cpp; do \
			if [ -f "$$example" ]; then \
				name=$$(basename $$example .cpp); \
				echo "Building example: $$name"; \
				if $(CXX) $(CXXFLAGS) $(INCLUDE_FLAGS) $$example -L$(BUILD_DIR) -l$(PROJECT_NAME) -o $(BUILD_DIR)/examples/$$name; then \
					echo "✓ Built example: $$name"; \
				else \
					echo "❌ Failed to build example: $$name"; \
					EXAMPLE_SUCCESS=false; \
				fi; \
			fi; \
		done; \
		if [ "$$EXAMPLE_SUCCESS" = "true" ]; then \
			echo "✅ Examples built successfully"; \
		else \
			echo "❌ Some examples failed to build"; \
			exit 1; \
		fi; \
	else \
		echo "ℹ️  No examples directory found"; \
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
				echo "Building sample: $$name"; \
				if $(CXX) $(CXXFLAGS) $(INCLUDE_FLAGS) $$sample $(CPP_FILES) -o $(BUILD_DIR)/samples/$$name; then \
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
	@if [ -d "samples" ]; then \
		echo "Building additional samples..."; \
		mkdir -p $(BUILD_DIR)/samples; \
		ADDITIONAL_SUCCESS=true; \
		for sample in samples/*.cpp; do \
			if [ -f "$$sample" ]; then \
				name=$$(basename $$sample .cpp); \
				echo "Building sample: $$name"; \
				if $(CXX) $(CXXFLAGS) $(INCLUDE_FLAGS) $$sample $(CPP_FILES) -o $(BUILD_DIR)/samples/$$name; then \
					echo "✓ Built sample: $$name"; \
				else \
					echo "❌ Failed to build sample: $$name"; \
					ADDITIONAL_SUCCESS=false; \
				fi; \
			fi; \
		done; \
		if [ "$$ADDITIONAL_SUCCESS" = "true" ]; then \
			echo "✅ Additional samples built successfully"; \
		else \
			echo "❌ Some additional samples failed to build"; \
			exit 1; \
		fi; \
	fi

# Build and run XOR sample
.PHONY: xor
xor: samples
	@if [ -f "$(BUILD_DIR)/samples/xor" ]; then \
		echo "Running XOR sample..."; \
		$(BUILD_DIR)/samples/xor; \
	else \
		echo "❌ XOR sample not found"; \
	fi

# Build and run Model I/O test
.PHONY: model-format-test
model-format-test: samples
	@if [ -f "$(BUILD_DIR)/samples/model_format_test" ]; then \
		echo "Running Model Format test..."; \
		$(BUILD_DIR)/samples/model_format_test; \
	else \
		echo "❌ Model Format test not found"; \
	fi

# Show help
.PHONY: help
help:
	@echo "$(PROJECT_NAME) v$(VERSION) - Available targets:"
	@echo ""
	@echo "Building:"
	@echo "  all          - Build the library (default)"
	@echo "  debug        - Build with debug flags"
	@echo "  clean        - Remove build artifacts, training outputs, and test files"
	@echo "  deep-clean   - Remove all generated files including nested training dirs"
	@echo ""
	@echo "Testing:"
	@echo "  test         - Run all tests (unit + integration)"
	@echo "  unit-test    - Run unit tests only"
	@echo "  integration-test - Run integration tests only"
	@echo "  test-all     - Run comprehensive test runner"
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
	@echo "Examples & Samples:"
	@echo "  examples     - Build example programs"
	@echo "  samples      - Build sample programs"
	@echo "  xor          - Build and run XOR sample"
	@echo "  model-format-test - Build and run Model Format test"
	@echo ""
	@echo "Misc:"
	@echo "  help         - Show this help message"