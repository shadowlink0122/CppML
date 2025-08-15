# CUDA configuration for MLLib
# This file provides CUDA-specific build settings

# Check for CUDA installation
NVCC := $(shell which nvcc 2>/dev/null)
CUDA_HOME := $(shell dirname $(shell dirname $(NVCC)) 2>/dev/null)

# CUDA compiler and flags
NVCC_FLAGS = -std=c++17 -O2 --compiler-options -fPIC
CUDA_INCLUDE = -I$(CUDA_HOME)/include
CUDA_LIBS = -L$(CUDA_HOME)/lib64 -lcudart -lcublas

# CUDA source files
CUDA_FILES = $(shell find $(SRC_DIR) -name "*.cu" 2>/dev/null || true)

# Check if CUDA is available
ifeq ($(NVCC),)
    CUDA_AVAILABLE = false
    CUDA_MESSAGE = "CUDA not found - building CPU-only version"
else
    CUDA_AVAILABLE = true
    CUDA_MESSAGE = "CUDA found at $(CUDA_HOME) - building with GPU support"
endif

# CUDA compilation function
define compile_cuda
	@if [ "$(CUDA_AVAILABLE)" = "true" ]; then \
		echo "Compiling CUDA files..."; \
		for cu_file in $(CUDA_FILES); do \
			obj_file=$$(basename $$cu_file .cu).o; \
			echo "  nvcc $$cu_file -> $$obj_file"; \
			$(NVCC) $(NVCC_FLAGS) $(INCLUDE_FLAGS) $(CUDA_INCLUDE) -c $$cu_file -o $$obj_file; \
		done; \
	fi
endef
