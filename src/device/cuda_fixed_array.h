#pragma once

#include <cuda_runtime_api.h>

#include "common.h"


/**
 * Compile-time sized array with an operator[] usable from constexpr, host, and device.
 * Thin replacement for std::array (which isn't fully constexpr/device-friendly everywhere).
 */
template<typename T, uint8_t N>
struct CudaFixedArray
{
    static constexpr uint8_t Size = N;

    template<typename TIndex>
    __host__ __device__ __forceinline__ constexpr T& operator[](TIndex index) { return data[static_cast<uint8_t>(index)]; }

    template<typename TIndex>
    __host__ __device__ __forceinline__ constexpr const T& operator[](TIndex index) const { return data[static_cast<uint8_t>(index)]; }

    __host__ __device__ __forceinline__ constexpr uint8_t size() const { return Size; }

    __host__ __device__ __forceinline__ constexpr bool operator==(const CudaFixedArray<T, N>& other) const { return memcmp(data, other.data, Size * sizeof(T)) == 0; }

    T data[N];

public:
    __host__ __device__ __inline__ constexpr auto begin() const { return &data[0]; }

    __host__ __device__ __inline__ constexpr auto end() const { return &data[N]; }

public:
    /** Copy the payload of `src` into a `__constant__` memory symbol `target`. */
    inline static bool constantCopy(CudaFixedArray& target, const CudaFixedArray& src)
    {
        auto error = cudaMemcpyToSymbol(target.data, src.data, sizeof(CudaFixedArray));
        CUDA_CHECK(error);

        return error == cudaSuccess;
    }
};
