#pragma once

#include "common.h"

#include <vector>
#include <cuda_runtime_api.h>


/**
 *  A helper class for arrays allocated in GPU memory
 */
template<typename T>
struct CudaArray
{
    using TCudaArray = CudaArray<T>;

    T* CUDA_data;

    size_t num;

    __host__ __device__ inline T& operator[](uint32_t index)
    {
        assert(index < num && "Index is out of range in CudaArray");
        return CUDA_data[index];
    }
    __host__ __device__ inline const T& operator[](uint32_t index) const
    {
        assert(index < num && "Index is out of range in CudaArray");
        return CUDA_data[index];
    }

    // Frees all associated resources assuming this pointer is GPU address
    inline void free()
    {
        TCudaArray::free(this);
    }

    // Copies bytes from @source to this array
    // Assumes array was allocated on GPU, will not fail if num > size
    inline void write(const T* source, size_t num)
    {
        TCudaArray::write(this, source, num);
    }

    // Copies data from GPU to @target array
    inline size_t copy(std::vector<T>& target)
    {
        return TCudaArray::copy(this, target);
    }

    // Copies self GPU object (without all underlying data in array)
    // into CPU memory
    inline TCudaArray host()
    {
        // thiscall should work even with invalid pointer
        return TCudaArray::host(this);
    }

    // Copies this pointer (which assumes to be GPU) to host and return copy of the last element
    inline T host_last()
    {
        // thiscall should work even with invalid pointer
        TCudaArray HOST_array = TCudaArray::host(this);
        return TCudaArray::read(HOST_array, HOST_array.num - 1);
    }

public:
    static TCudaArray* allocate(const std::vector<T>& source);
    static void free(TCudaArray* array);

    static TCudaArray host(const TCudaArray* device);

    static T read(const TCudaArray& HOST_Array, size_t index);

    static size_t copy(const TCudaArray* array, std::vector<T>& target);

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
CudaArray<T>* CudaArray<T>::allocate(const std::vector<T>& source)
{
    // Allocate memory of vector itself (ptr + size_t == 16 bytes)
    CudaArray<T>* result = nullptr;
    uint32_t error = cudaMalloc((void**) & result, sizeof(CudaArray<T>));
    CUDA_CHECK(error);

    // Write size_t - size of the data
    size_t source_size = source.size();
    error = cudaMemcpy(&result->num, &source_size, sizeof(size_t), cudaMemcpyHostToDevice);
    CUDA_CHECK(error);

    // Device pointer of data (if available)
    T* data_ptr = nullptr;
    if (source_size > 0)
    {
        size_t allocated_bytes = sizeof(T) * source_size;

        // allocate data on device and copy from RAW
        error = cudaMalloc((void**)&data_ptr, allocated_bytes);
        CUDA_CHECK(error);

        error = cudaMemcpy(data_ptr, source.data(), allocated_bytes, cudaMemcpyHostToDevice);
        CUDA_CHECK(error);
    }

    // Write data pointer (null if no data)
    error = cudaMemcpy(&result->CUDA_data, &data_ptr, sizeof(T*), cudaMemcpyHostToDevice);
    CUDA_CHECK(error);

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
        target.reserve(HOST_array.num);

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