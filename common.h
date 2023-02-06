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


#define NO_INNER_LOOPS 1


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


namespace str
{

template<typename String, typename ... Args>
inline String format(const String& format, Args ... args)
{
	int size_s = snprintf(nullptr, 0, format.c_str(), args ...) + 1; // Extra space for '\0'
	if (size_s <= 0)
	{
		return {};
	}

	auto size = static_cast<size_t>(size_s);
	String result(size, '\0');

	snprintf(result.data(), size, format.c_str(), args ...);
	return result; // Extra '\0' at the end
}

}