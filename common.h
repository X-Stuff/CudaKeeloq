#pragma once

#include "stdint.h"

#if defined(__CUDA_ARCH__) || defined(CU_FILE)
	#define LOCATION	 /* C++ compiler via NVCC - should be global NS also */
#else
	#define LOCATION CPU /* C++ compiler */
#endif


#if defined(__CUDA_ARCH__) || defined(CU_FILE)
	/* .cu files should be all in global namespace */

	#define NS_LOCATION_BEGIN
	#define NS_LOCATION_END

	#define NS_LOCATION
	#define USE_NS_LOCATION
#else
	#define NS_LOCATION namespace LOCATION
	#define USE_NS_LOCATION using namespace LOCATION;

	#define NS_LOCATION_BEGIN NS_LOCATION {
	#define NS_LOCATION_END }
#endif


#include <assert.h>

#define CUDA_CHECK(error) \
    if (error != 0) { printf("\nASSERTION FAILED. CUDA ERROR!\n%s: %s\n", cudaGetErrorName((cudaError_t)error), cudaGetErrorString((cudaError_t)error)); }\
    assert(error == 0)


#define GENERATOR_KERNEL_NAME(name) \
	Kernel_##name

#define GENERATOR_KERNEL_GETTER_NAME(name) \
	GetKernel_##name

#define DEFINE_GENERATOR_KERNEL(name, ...) \
	GENERATOR_KERNEL_NAME(name)(__VA_ARGS__)

#define DEFINE_GENERATOR_GETTER(name) \
	extern "C" void* GENERATOR_KERNEL_GETTER_NAME(name)() { return &Kernel_##name; }
