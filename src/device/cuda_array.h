#pragma once

#include <vector>
#include <limits>

#include <cuda_runtime_api.h>

#include "common.h"


/**
 * Owning array allocated in GPU memory.
 * The struct itself also lives in device memory — host-side helpers are static and take `device`
 * pointers; member helpers (free/write/copy/read/host) dispatch to those statics.
 */
template<typename T>
struct CudaArray
{
    using TCudaArray = CudaArray<T>;

    T* CUDA_data;

    size_t num;

    __host__ __device__ inline T& operator[](size_t index)
    {
        assert(index < num && "Index is out of range in CudaArray");
        return CUDA_data[index];
    }
    __host__ __device__ inline const T& operator[](size_t index) const
    {
        assert(index < num && "Index is out of range in CudaArray");
        return CUDA_data[index];
    }

    /** Release the underlying GPU allocation (this pointer must be a device address). */
    inline void free()
    {
        TCudaArray::free(this);
    }

    /** Upload `num` elements from host memory into this array. */
    inline void write(const T* source, size_t num)
    {
        TCudaArray::write(this, source, num);
    }

    /** Copy the full array from GPU into `target`, returning the element count. */
    inline size_t copy(std::vector<T>& target) const
    {
        return TCudaArray::copy(this, target);
    }

    /** Download a slice (up to full array) `[index, index+num)` from GPU into a new host vector. */
    inline std::vector<T> read(size_t index = 0, size_t num = std::numeric_limits<size_t>::max()) const
    {
        return TCudaArray::read(host(), index, num);
    }

    /** Copy this array's header (pointer + size) from GPU into host memory. */
    inline TCudaArray host() const
    {
        // thiscall should work even with invalid pointer
        return TCudaArray::host(this);
    }

    /** Total device bytes allocated for this array's payload. */
    inline size_t allocated() const
    {
        // thiscall should work even with invalid pointer
        return TCudaArray::host(this).num * sizeof(T);
    }

    /** Download the last element of this array from GPU. */
    inline T hostLast() const
    {
        // thiscall should work even with invalid pointer
        TCudaArray HOST_array = TCudaArray::host(this);
        return TCudaArray::read(HOST_array, HOST_array.num - 1);
    }

public:
    /** Allocate a device array seeded with `source`'s contents. */
    static TCudaArray* allocate(const std::vector<T>& source);

    /** Allocate an uninitialised device array of `size` elements. */
    static TCudaArray* allocate(const size_t size);

    /** Free a device array (its header and payload). */
    static void free(TCudaArray* array);

    /** Copy a device array's header (pointer + size) back to host. */
    static TCudaArray host(const TCudaArray* device);

    /** Download a single element from a host-side header's payload. */
    static T read(const TCudaArray& HOST_Array, size_t index);

    /** Download a slice (up to full array) from a host-side header's payload. */
    static std::vector<T> read(const TCudaArray& HOST_Array, size_t index, size_t num);

    /** Download an entire array into the supplied host vector. */
    static size_t copy(const TCudaArray* array, std::vector<T>& target);

    /** Upload `num` elements from host memory into a device array. */
    static void write(TCudaArray* dest, const T* source, size_t num);
};

template<typename T>
T CudaArray<T>::read(const TCudaArray& HOST_Array, size_t index)
{
    assert(index < HOST_Array.num);

    T result;
    auto error = cudaMemcpy(&result, &HOST_Array.CUDA_data[index], sizeof(T), cudaMemcpyDeviceToHost);
    CUDA_CHECK(error);

    return result;
}

template<typename T>
std::vector<T> CudaArray<T>::read(const TCudaArray& HOST_Array, size_t index, size_t num)
{
    assert(index < HOST_Array.num);

    auto elements = std::min(num, HOST_Array.num - index);

    std::vector<T> result(elements);
    auto error = cudaMemcpy(result.data(), &HOST_Array.CUDA_data[index], elements * sizeof(T), cudaMemcpyDeviceToHost);
    CUDA_CHECK(error);

    return result;
}

template<typename T>
CudaArray<T>* CudaArray<T>::allocate(const size_t size)
{
    // Allocate memory of vector itself (ptr + size_t == 16 bytes)
    CudaArray<T>* result = nullptr;
    uint32_t error = cudaMalloc((void**)&result, sizeof(CudaArray<T>));
    CUDA_CHECK_RETURN(error, nullptr);

    // Write size_t - size of the data
    error = cudaMemcpy(&result->num, &size, sizeof(size_t), cudaMemcpyHostToDevice);
    CUDA_CHECK_RETURN(error, nullptr);

    // Device pointer of data (if available)
    T* data_ptr = nullptr;
    if (size > 0)
    {
        const size_t allocated_bytes = sizeof(T) * size;

        // allocate data on device and copy from RAW
        error = cudaMalloc((void**)&data_ptr, allocated_bytes);
        CUDA_CHECK_RETURN(error, nullptr);
    }

    // Write data pointer (null if no data)
    error = cudaMemcpy(&result->CUDA_data, &data_ptr, sizeof(T*), cudaMemcpyHostToDevice);
    CUDA_CHECK_RETURN(error, nullptr);

    return result;
}

template<typename T>
CudaArray<T>* CudaArray<T>::allocate(const std::vector<T>& source)
{
    auto result = allocate(source.size());
    if (result && source.size() > 0)
    {
        const size_t allocated_bytes = sizeof(T) * source.size();

        T* data_ptr = nullptr;
        cudaMemcpy(&data_ptr, &result->CUDA_data, sizeof(T*), cudaMemcpyDeviceToHost);

        assert(data_ptr && "CUDA Data wasn't allocated");

        // Write data pointer (null if no data)
        auto error = cudaMemcpy(data_ptr, source.data(), allocated_bytes, cudaMemcpyHostToDevice);
        CUDA_CHECK(error);
    }

    return result;
}

template<typename T>
CudaArray<T> CudaArray<T>::host(const TCudaArray* device)
{
    assert(device && "Invalid CUDA pointer");
    TCudaArray HOST_dest = { 0 };

    auto error = cudaMemcpy(&HOST_dest, device, sizeof(TCudaArray), cudaMemcpyDeviceToHost);
    CUDA_CHECK(error);

    return HOST_dest;
}

template<typename T>
void CudaArray<T>::free(TCudaArray* array)
{
    TCudaArray HOST_array = host(array);

    if (HOST_array.CUDA_data)
    {
        auto error = cudaFree(HOST_array.CUDA_data);
        CUDA_CHECK(error);
    }

    auto error = cudaFree(array);
    CUDA_CHECK(error);
}

template<typename T>
size_t CudaArray<T>::copy(const TCudaArray* array, std::vector<T>& target)
{
    TCudaArray HOST_array = host(array);
    if (HOST_array.num > 0)
    {
        size_t allocated_bytes = HOST_array.num * sizeof(T);
        target.resize(HOST_array.num);

        auto error = cudaMemcpy((T*)target.data(), HOST_array.CUDA_data, allocated_bytes, cudaMemcpyDeviceToHost);
        CUDA_CHECK(error);
    }

    return HOST_array.num;
}

template<typename T>
void CudaArray<T>::write(TCudaArray* dest, const T* source, size_t num)
{
    TCudaArray HOST_dest = host(dest);

    assert(HOST_dest.CUDA_data && "CUDA Data wasn't allocated");
    if (HOST_dest.CUDA_data)
    {
        auto copy_bytes = sizeof(T) * std::min(num, HOST_dest.num);

        auto error = cudaMemcpy(HOST_dest.CUDA_data, source, copy_bytes, cudaMemcpyHostToDevice);
        CUDA_CHECK(error);
    }
}
