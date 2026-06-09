#pragma once

#include <vector>

#include <cuda_runtime_api.h>

#include "common.h"

#include "device/cuda_array.h"


/**
 * Host-owned std::vector with a lazily-allocated GPU mirror.
 * The host copy is considered the source of truth; `read()` pulls the latest GPU state back.
 */
template<typename T>
struct CudaVector
{
    /** Construct from an initializer list. */
    CudaVector(std::initializer_list<T>&& initializer) :
        cpu_vector(std::move(initializer)),
        gpu_array(nullptr)
    {
    }

    /** Forward any remaining args to the underlying std::vector. */
    template<typename ... Args>
    CudaVector(Args&&... args) :
        cpu_vector(std::forward<Args>(args)...),
        gpu_array(nullptr)
    {
    }

    ~CudaVector();

    /** Immutable view of the host data (mutations happen through kernel → `read()`). */
    const std::vector<T>& cpu() const { return cpu_vector; }

    /** Element count (matches GPU size once the mirror is allocated). */
    const size_t size() const { return cpu_vector.size(); }

    /** Copy GPU data back into the host vector. Returns `*this`. */
    CudaVector<T>& read();

    /** GPU mirror pointer; allocates (and uploads) on the first call by default. */
    CudaArray<T>* gpu(bool allocate = true);

private:

    std::vector<T> cpu_vector;

    CudaArray<T>* gpu_array;
};

template<typename T> CudaVector<T>& CudaVector<T>::read()
{
    assert(gpu_array);
    if (gpu_array)
    {
        gpu_array->copy(cpu_vector);
    }

    return *this;
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
