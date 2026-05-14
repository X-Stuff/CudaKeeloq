# Linux/GCC makefile for CudaKeeloq.
# Targets CUDA 13.2 (nvidia/cuda:13.2.0-devel-ubuntu22.04 in docker).
#
# Two binaries are produced from a single source tree:
#   CudaKeeloq       — main application (entry point: src/main.cpp)
#   CudaKeeloqTests  — doctest runner   (entry point: src/tests/test_main.cpp
#                                        + all src/tests/*.{cpp,cu} files)
#
# Usage:
#   make              → debug, both binaries
#   make release      → release, both binaries
#   make profile      → profile, both binaries
#   make app          → build only the app (debug unless CONFIG= is set)
#   make tests        → build only the tests
#   make clean        → wipe x64/
#
# Override config explicitly:  make CONFIG=release
# Override CUDA toolkit path:   make CUDA_ROOT_DIR=/opt/cuda
#
# The tests binary supports doctest filters, e.g.:
#   ./CudaKeeloqTests --test-case="*filter*"

TARGET_APP   := CudaKeeloq
TARGET_TESTS := CudaKeeloqTests

CUDA_ROOT_DIR ?= /usr/local/cuda
ARCH          := x64

# ----------------------------------------------------------------------
# Config selection
# ----------------------------------------------------------------------
# Goals {release, profile, debug} act as config *aliases*. They all funnel
# into the `all` target below; the variable CONFIG determines OBJ_DIR and
# flags. This way OBJ_DIR is known at parse time, so pattern-rule targets
# like `$(OBJ_DIR)/%.o` expand correctly.

ifneq (,$(filter release,$(MAKECMDGOALS)))
  CONFIG ?= release
else ifneq (,$(filter profile,$(MAKECMDGOALS)))
  CONFIG ?= profile
else ifneq (,$(filter debug,$(MAKECMDGOALS)))
  CONFIG ?= debug
else
  CONFIG ?= debug
endif

OBJ_DIR := ./.build/$(ARCH)/$(CONFIG)/linux/obj
EXE_DIR := ./.build/$(ARCH)/$(CONFIG)/linux/bin

# ----------------------------------------------------------------------
# Compilers + flags
# ----------------------------------------------------------------------

CC        := g++
CC_FLAGS  := -std=c++17 -Wall
CC_INC    := -I./src/ -I./ThirdParty/ -I./ThirdParty/cpp-terminal

NVCC       := nvcc
# Default target arch: sm_86. Release/Profile override this with an LTO arch so
# the -dlto link step has something to work with (CUDA 13.2 requires
# `code=lto_<arch>` as a first-class code kind, not the old `-dlto` compile flag).
NVCC_ARCH  := -arch=sm_86
NVCC_FLAGS := $(NVCC_ARCH) --std=c++17 -rdc=true
NVCC_INC   := -I./src/ -I./ThirdParty/

CUDA_LIB_DIR := -L$(CUDA_ROOT_DIR)/lib64
CUDA_INC_DIR := -I$(CUDA_ROOT_DIR)/include
CUDA_LINK    := -lcudart -lcudadevrt

ifeq ($(CONFIG),release)
  # Replace plain sm_86 with LTO code kind; nvlink's -dlto will produce real SASS.
  # Device-link still needs a physical target (-arch=sm_86) to emit runnable code.
  NVCC_FLAGS := $(filter-out -arch=sm_86,$(NVCC_FLAGS)) \
                -gencode=arch=compute_86,code=lto_86 \
                -use_fast_math -O3 -Xptxas -O3 --m64
  NVLINK_FLAGS := -dlto -arch=sm_86
  CC_FLAGS     += -O3 -DNDEBUG
else ifeq ($(CONFIG),profile)
  NVCC_FLAGS := $(filter-out -arch=sm_86,$(NVCC_FLAGS)) \
                -gencode=arch=compute_86,code=lto_86 \
                -lineinfo -use_fast_math
  NVLINK_FLAGS := -dlto -arch=sm_86
  CC_FLAGS     += -DNDEBUG
else ifeq ($(CONFIG),debug)
  NVCC_FLAGS += -G
  NVLINK_FLAGS := $(NVCC_ARCH)
  CC_FLAGS   += -D_DEBUG
else
  $(error Unknown CONFIG=$(CONFIG). Use one of: debug, release, profile)
endif

# ----------------------------------------------------------------------
# Source sets
# ----------------------------------------------------------------------

# Shared C++ sources: everything under src/ except main.cpp and src/tests/
SHARED_CPP_FILES := $(shell find src/ -name '*.cpp' \
                         -not -path 'src/tests/*' \
                         -not -path 'src/main.cpp')

APP_CPP_FILES    := src/main.cpp
TEST_CPP_FILES   := $(shell find src/tests -name '*.cpp')

# Shared CUDA kernels used by both binaries. test_kernel.cu is tests-only.
SHARED_CU_FILES  := $(shell find src/ -name '*.cu' -not -path 'src/tests/*')
TEST_CU_FILES    := src/tests/test_kernel.cu

# Object-file layout mirrors source paths under $(OBJ_DIR)/ so two files with
# the same basename in different folders do not collide.
SHARED_CPP_OBJS := $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(SHARED_CPP_FILES))
SHARED_CU_OBJS  := $(patsubst %.cu,$(OBJ_DIR)/%.o,$(SHARED_CU_FILES))

APP_CPP_OBJS    := $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(APP_CPP_FILES))

TEST_CPP_OBJS   := $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(TEST_CPP_FILES))
TEST_CU_OBJS    := $(patsubst %.cu,$(OBJ_DIR)/%.o,$(TEST_CU_FILES))

# ----------------------------------------------------------------------
# Top-level targets
# ----------------------------------------------------------------------

.PHONY: all app tests release profile debug clean

all: app tests

# Config aliases — they just delegate to `all`. CONFIG was already selected
# above from MAKECMDGOALS, so no variable plumbing is needed here.
release: all
profile: all
debug:   all

app:   $(EXE_DIR)/$(TARGET_APP)
tests: $(EXE_DIR)/$(TARGET_TESTS)

# Device-link is per-binary because it must see every .cu object that will be
# linked into the final executable (-rdc=true).

$(EXE_DIR)/$(TARGET_APP): $(SHARED_CPP_OBJS) $(SHARED_CU_OBJS) $(APP_CPP_OBJS)
	@mkdir -p $(EXE_DIR)
	$(NVCC) $(NVLINK_FLAGS) --std=c++17 -rdc=true $(NVCC_INC) -dlink \
		$(SHARED_CU_OBJS) \
		-o $(OBJ_DIR)/device_link_app.o
	$(CC) $(CC_FLAGS) \
		$(SHARED_CPP_OBJS) $(SHARED_CU_OBJS) $(APP_CPP_OBJS) \
		$(OBJ_DIR)/device_link_app.o \
		-o $@ $(CUDA_INC_DIR) $(CUDA_LIB_DIR) $(CUDA_LINK)

$(EXE_DIR)/$(TARGET_TESTS): $(SHARED_CPP_OBJS) $(SHARED_CU_OBJS) $(TEST_CPP_OBJS) $(TEST_CU_OBJS)
	@mkdir -p $(EXE_DIR)
	$(NVCC) $(NVLINK_FLAGS) --std=c++17 -rdc=true $(NVCC_INC) -dlink \
		$(SHARED_CU_OBJS) $(TEST_CU_OBJS) \
		-o $(OBJ_DIR)/device_link_tests.o
	$(CC) $(CC_FLAGS) \
		$(SHARED_CPP_OBJS) $(SHARED_CU_OBJS) $(TEST_CPP_OBJS) $(TEST_CU_OBJS) \
		$(OBJ_DIR)/device_link_tests.o \
		-o $@ $(CUDA_INC_DIR) $(CUDA_LIB_DIR) $(CUDA_LINK)

# ----------------------------------------------------------------------
# Compile rules
# ----------------------------------------------------------------------

$(OBJ_DIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CC) $(CC_FLAGS) $(CC_INC) $(CUDA_INC_DIR) -c $< -o $@

$(OBJ_DIR)/%.o: %.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCC_FLAGS) $(NVCC_INC) -c $< -o $@

clean:
	rm -rf $(OBJ_DIR)
	rm -rf $(EXE_DIR)
