# THIS MAKE FILE IS SUPPOSED TO BE USED IN WITH DOCKER IMAGE nvidia/cuda:12.0.1-devel

TARGET_NAME=CudaKeeloq

CUDA_ROOT_DIR=/usr/local/cuda

ARCH=x64
CONFIG_RELEASE=release
CONFIG_PROFILE=profile
CONFIG_DEBUG=debug

# CC compiler options:
CC=g++
CC_FLAGS=-std=c++17 -Wall
CC_INCLUDE=-I./src/ -I./ThirdParty/ -I./ThirdParty/cpp-terminal

NVCC=nvcc
NVCC_FLAGS=-arch=sm_86 --std=c++17 -rdc=true
NVCC_INCLUDE=-I./src/

# CUDA library directory:
CUDA_LIB_DIR= -L$(CUDA_ROOT_DIR)/lib64
# CUDA include directory:
CUDA_INC_DIR= -I$(CUDA_ROOT_DIR)/include
# CUDA linking libraries:
CUDA_LINK_LIBS= -lcudart -lcudadevrt


# Configurations, default debug
all: debug
	@echo No target specified. Default is debug

release: OBJ_DIR=./$(ARCH)/$(CONFIG_RELEASE)/obj
release: EXE_DIR=./$(ARCH)/$(CONFIG_RELEASE)/bin
release: NVCC_FLAGS+= -use_fast_math -O3 -Xptxas -O3 --m64 -dlto
release: CC_FLAGS+= -O3 -DNDEBUG
release:link

profile: OBJ_DIR=./$(ARCH)/$(CONFIG_PROFILE)/obj
profile: EXE_DIR=./$(ARCH)/$(CONFIG_PROFILE)/bin
profile: NVCC_FLAGS+= -lineinfo -use_fast_math -dlto
profile: CC_FLAGS+= -DNDEBUG
profile:link

debug:   OBJ_DIR=./$(ARCH)/$(CONFIG_DEBUG)/obj
debug:   EXE_DIR=./$(ARCH)/$(CONFIG_DEBUG)/bin
debug:   NVCC_FLAGS+= -G
debug:   CC_FLAGS+= -D_DEBUG
debug:link

# Sources C++
CPP_FILES = $(shell find src/ -iname "*.cpp")
CPP_OBJECTS = $(CPP_FILES:%.cpp=%.o)

# Sources CUDA
CUDA_FILES = $(shell find src/ -iname "*.cu")
CUDA_OBJECTS = $(CUDA_FILES:%.cu=%.o)

# Device link (required because we compile with -rdc=true)
dlink: $(CUDA_OBJECTS)
	$(NVCC) $(NVCC_FLAGS) $(NVCC_INCLUDE) -dlink \
		$(addprefix $(OBJ_DIR)/, $(notdir $(CUDA_OBJECTS))) \
		-o $(OBJ_DIR)/device_link.o

# Link
link: $(CPP_OBJECTS) $(CUDA_OBJECTS) dlink
	$(CC) $(CC_FLAGS) \
		$(addprefix $(OBJ_DIR)/, $(notdir $(CPP_OBJECTS))) \
		$(addprefix $(OBJ_DIR)/, $(notdir $(CUDA_OBJECTS))) \
		$(OBJ_DIR)/device_link.o \
		-o $(EXE_DIR)/$(TARGET_NAME) $(CUDA_INC_DIR) $(CUDA_LIB_DIR) $(CUDA_LINK_LIBS)

# Compile C++
$(CPP_OBJECTS): | mkdirs
	$(CC) $(CC_FLAGS) $(CC_INCLUDE) $(CUDA_INC_DIR) -c $(basename $@).cpp -o $(OBJ_DIR)/$(notdir $@)

# Compile CUDA
$(CUDA_OBJECTS): | mkdirs
	$(NVCC) $(NVCC_FLAGS) $(NVCC_INCLUDE) -c $(basename $@).cu -o $(OBJ_DIR)/$(notdir $@)

# Prepare
mkdirs:
	mkdir -p $(OBJ_DIR)
	mkdir -p $(EXE_DIR)
