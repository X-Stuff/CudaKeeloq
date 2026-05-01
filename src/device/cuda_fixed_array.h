#pragma once

#include "common.h"

#include <cuda_runtime_api.h>

/**
 *  Fixed array - allows to be used in constexpr function and have operator[] for both host and device code. Size is template parameter.
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
    // Number of allocated bytes in GPU memory for this array
    inline static bool constant_copy(CudaFixedArray& target, const CudaFixedArray& src)
    {
        auto error = cudaMemcpyToSymbol(target.data, src.data, sizeof(CudaFixedArray));
        CUDA_CHECK(error);

        return error == cudaSuccess;
    }
};
