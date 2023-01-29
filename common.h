#pragma once

#include <stdint.h>
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


#if __CUDA_ARCH__
	#define UNROLL #pragma unroll
#else
	#define UNROLL
#endif



template<uint8_t NSize = 255>
struct DefaultByteArray
{
	uint8_t element[NSize];

	constexpr DefaultByteArray() : element()
	{
		for (uint8_t i = 0; i < NSize; ++i)
		{
			element[i] = i;
		}
	}

	template<typename Vector>
	static inline Vector as_vector()
	{
		constexpr DefaultByteArray array = DefaultByteArray();
		return Vector(&array.element[0], &array.element[0] + NSize);
	}
};
