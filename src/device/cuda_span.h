#pragma once

#include "common.h"

#include <type_traits>
#include <cuda_runtime_api.h>


/**
 *  Something like C# span
 * Memory view
 */
template<typename T>
struct Span
{
    __host__ __device__ Span(T* ptr, uint32_t num) : data(ptr), size(num) { }

    __host__ __device__ inline T& operator[](uint32_t index)
    {
        assert(index < size && "Index is out of range in Span");

#if __CUDA_ARCH__
        return __ldca(&data[index]);
#else
        return data[index];
#endif
    }

    __host__ __device__ inline const T& operator[](uint32_t index) const
    {
        assert(index < size && "Index is out of range in Span");
#if __CUDA_ARCH__
        return __ldca(&data[index]);
#else
        return data[index];
#endif
    }

    // Number of elements in Span
    __host__ __device__ inline uint32_t num() const { return size; }

    // fall back if T is not simple type
    __host__ __device__ typename std::enable_if<!std::is_integral_v<T>, T&>::type __ldca(T* ptr) { return *ptr; }
    __host__ __device__ typename std::enable_if<!std::is_integral_v<T>, const T&>::type __ldca(T* ptr) const { return *ptr; }

private:

    T* data;

    uint32_t size;
};