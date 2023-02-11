#pragma once

#include "common.h"

#include "device/cuda_array.h"

#include <vector>
#include <cuda_runtime_api.h>


/**
 *  Host owned, GPU copy, vector
 * Can be considered as wrapper around host vector
 * Owns GPU data, frees on destructor
 */
template<typename T>
struct CudaVector
{
    // Explicit construct using initializer list
    CudaVector(std::initializer_list<T>&& initializer) :
        cpu_vector(std::move(initializer)),
        gpu_array(nullptr)
    {
    }

    // Forward construction to vector
    template<typename ... Args>
    CudaVector(Args&&... args) :
        cpu_vector(std::forward<Args>(args)...),
        gpu_array(nullptr)
    {
    }

    ~CudaVector();

    // CPU memory is immutable since it should be
    // synchronized with gpu
    const std::vector<T>& cpu() const { return cpu_vector; }

    // Size of CPU vector (gpu will be the same size once allocated)
    const size_t size() const { return cpu_vector.size(); }

    // Reads GPU pointer and copies data from GPU memory to CPU
    void read();

    //  Pointer to GPU array
    // Will allocate (and copy) by default
    CudaArray<T>* gpu(bool allocate = true);

private:

    std::vector<T> cpu_vector;

    CudaArray<T>* gpu_array;
};

template<typename T> void CudaVector<T>::read()
{
    assert(gpu_array);
    if (gpu_array)
    {
        gpu_array->copy(cpu_vector);
    }
}

template<typename T> CudaArray<T>* CudaVector<T>::gpu(bool allocate /*= true*/)
{
    if (gpu_array || !allocate)
    {
        return gpu_array;
    }

    gpu_array = CudaArray<T>::allocate(cpu_vector);
    return gpu_array;
}

template<typename T> CudaVector<T>::~CudaVector()
{
    if (gpu_array != nullptr)
    {
        gpu_array->free();
        gpu_array = nullptr;
    }
}
