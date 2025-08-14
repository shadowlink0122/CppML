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
SAMPLE_DIR = sample

# Compiler settings
CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2
DEBUG_FLAGS = -g -DDEBUG
INCLUDE_FLAGS = -I$(INCLUDE_DIR)

# Find source files
CPP_FILES = $(shell find $(SRC_DIR) -name "*.cpp" 2>/dev/null || true)
HPP_FILES = $(shell find $(INCLUDE_DIR) -name "*.hpp" 2>/dev/null || true)
TEST_FILES = $(shell find $(TEST_DIR) -name "*.cpp" 2>/dev/null || true)

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
	@if [ -n "$(CPP_FILES)" ]; then \
		$(CXX) $(CXXFLAGS) $(INCLUDE_FLAGS) -c $(CPP_FILES); \
		ar rcs $(LIB_TARGET) *.o; \
		rm -f *.o; \
		echo "✅ Library built successfully: $(LIB_TARGET)"; \
	else \
		echo "ℹ️  No source files found, creating header-only library marker"; \
		touch $(LIB_TARGET); \
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
	@echo "✅ Clean completed"

# Deep clean - remove all generated files
.PHONY: deep-clean
deep-clean: clean
	@echo "Performing deep clean..."
	@rm -rf .vscode/settings.json.backup
	@find . -name "*.tmp" -delete 2>/dev/null || true
	@find . -name "*.log" -delete 2>/dev/null || true
	@find . -name ".DS_Store" -delete 2>/dev/null || true
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

# Run tests (if test directory exists)
.PHONY: test
test: $(LIB_TARGET)
	@if [ -d "$(TEST_DIR)" ] && [ -n "$(TEST_FILES)" ]; then \
		echo "Building and running tests..."; \
		$(CXX) $(CXXFLAGS) $(INCLUDE_FLAGS) $(TEST_FILES) -L$(BUILD_DIR) -l$(PROJECT_NAME) -o $(BUILD_DIR)/test_runner; \
		$(BUILD_DIR)/test_runner; \
		echo "✅ Tests completed"; \
	else \
		echo "ℹ️  No tests found in $(TEST_DIR)"; \
	fi

# Build examples (if examples directory exists)
.PHONY: examples
examples: $(LIB_TARGET)
	@if [ -d "$(EXAMPLE_DIR)" ]; then \
		echo "Building examples..."; \
		mkdir -p $(BUILD_DIR)/examples; \
		for example in $(EXAMPLE_DIR)/*.cpp; do \
			if [ -f "$$example" ]; then \
				name=$$(basename $$example .cpp); \
				$(CXX) $(CXXFLAGS) $(INCLUDE_FLAGS) $$example -L$(BUILD_DIR) -l$(PROJECT_NAME) -o $(BUILD_DIR)/examples/$$name; \
				echo "Built example: $$name"; \
			fi; \
		done; \
		echo "✅ Examples built successfully"; \
	else \
		echo "ℹ️  No examples directory found"; \
	fi

# Build samples (if sample directory exists)
.PHONY: samples
samples: $(LIB_TARGET)
	@if [ -d "$(SAMPLE_DIR)" ]; then \
		echo "Building samples..."; \
		mkdir -p $(BUILD_DIR)/samples; \
		for sample in $(SAMPLE_DIR)/*.cpp; do \
			if [ -f "$$sample" ]; then \
				name=$$(basename $$sample .cpp); \
				echo "Building sample: $$name"; \
				$(CXX) $(CXXFLAGS) $(INCLUDE_FLAGS) $$sample $(CPP_FILES) -o $(BUILD_DIR)/samples/$$name; \
				echo "Built sample: $$name"; \
			fi; \
		done; \
		echo "✅ Samples built successfully"; \
	else \
		echo "ℹ️  No samples directory found"; \
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

# Show help
.PHONY: help
help:
	@echo "$(PROJECT_NAME) v$(VERSION) - Available targets:"
	@echo ""
	@echo "Building:"
	@echo "  all          - Build the library (default)"
	@echo "  debug        - Build with debug flags"
	@echo "  clean        - Remove build artifacts"
	@echo "  deep-clean   - Remove all generated files"
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
	@echo "Testing:"
	@echo "  test         - Build and run tests"
	@echo "  examples     - Build example programs"
	@echo "  samples      - Build sample programs"
	@echo "  xor          - Build and run XOR sample"
	@echo ""
	@echo "Misc:"
	@echo "  help         - Show this help message"