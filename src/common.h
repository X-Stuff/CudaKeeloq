#pragma once

#include <cstddef>
#include <stdint.h>
#include <assert.h>
#include <stdio.h>
#include <inttypes.h>


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
    extern "C" void* GENERATOR_KERNEL_GETTER_NAME(name)() { return (void*)&Kernel_##name; }


#if __CUDA_ARCH__
    #define UNROLL #pragma unroll
#else
    #define UNROLL
#endif

#ifndef NO_INNER_LOOPS
    #define NO_INNER_LOOPS 1
#endif // !NO_INNER_LOOPS



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

    snprintf(&result[0], size, format.c_str(), args ...);
    result.pop_back(); // remove '\0' from the end

    return result;
}

}


inline uint64_t operator "" _u64(const char* ascii, size_t num)
{
    const uint8_t size = sizeof(uint64_t);

    uint64_t number = 0;
    uint8_t* pNumber = (uint8_t*)&number;

    for (uint8_t i = 0; i < size; ++i)
    {
        pNumber[size - i - 1] = num > i ? ascii[i] : 0;
    }

    return number;
}