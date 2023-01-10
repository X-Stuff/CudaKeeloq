#pragma once

#include <vector>

#include "CUDA_check.cuh"

#ifndef __device__
#define __device__
#endif

#ifndef __global__
#define __global__
#endif

#ifndef __host__
#define __host__
#endif


#define GET_CUDA_CONTEXT() { \
                gridDim.x * blockDim.x,                /* thread_max */ \
                blockIdx.x * blockDim.x + threadIdx.x  /* thread_id  */ \
};


#define CUDA_FOR_THREAD_ID(ctx, i, count) for(uint32_t i = ctx.thread_id; i < count; i += ctx.thread_id)


//
struct CUDACtx
{
    uint32_t thread_max;
    uint32_t thread_id;
};


template<typename T>
struct CUDA_Array
{
    T* CUDA_data;
    size_t num;

    void free()
    {
        CUDA_Array<T>::free(this);
    }

    size_t copy(std::vector<T>& target)
    {
        return CUDA_Array<T>::copy(this, target);
    }

    static CUDA_Array<T>* allocate(const std::vector<T>& source)
    {
        // Allocate memory of vector itself (ptr + size_t == 16 bytes)
        CUDA_Array<T>* result = nullptr;
        uint32_t error = cudaMalloc(&result, sizeof(CUDA_Array<T>));
        CUDA_CHECK(error);

        // Write size_t - size of the data
        size_t source_size = source.size();
        error = cudaMemcpy(&result->num, &source_size, source_size, cudaMemcpyHostToDevice);
        CUDA_CHECK(error);

        // Device pointer of data (if available)
        T* data_ptr = nullptr;
        if (source_size > 0)
        {
            size_t allocated_bytes = sizeof(T) * source_size;

            // allocate data on device and copy from RAW
            error = cudaMalloc(&data_ptr, allocated_bytes);
            CUDA_CHECK(error);

            error = cudaMemcpy(data_ptr, source.data(), allocated_bytes, cudaMemcpyHostToDevice);
            CUDA_CHECK(error);
        }

        // Write data pointer (null if no data)
        error = cudaMemcpy(&result->CUDA_data, &data_ptr, sizeof(T*), cudaMemcpyHostToDevice);
        CUDA_CHECK(error);

        return result;
    }

    static void free(CUDA_Array<T>* vector)
    {
        T* data_ptr = nullptr;
        uint32_t error = cudaMemcpy(&data_ptr, &vector->CUDA_data, sizeof(T*), cudaMemcpyDeviceToHost);
        CUDA_CHECK(error);

        if (data_ptr)
        {
            error = cudaFree(data_ptr);
            CUDA_CHECK(error);
        }
        error = cudaFree(vector);
        CUDA_CHECK(error);
    }

    static size_t copy(const CUDA_Array<T>* vector, std::vector<T>& target)
    {
        CUDA_Array<T> device_vector;
        uint32_t error = cudaMemcpy(&device_vector, vector, sizeof(CUDA_Array<T>), cudaMemcpyDeviceToHost);
        CUDA_CHECK(error);

        if (device_vector.num > 0)
        {
            size_t allocated_bytes = device_vector.num * sizeof(T);
            target.reserve(allocated_bytes);

            error = cudaMemcpy((T*)target.data(), device_vector.CUDA_data, allocated_bytes, cudaMemcpyDeviceToHost);
            CUDA_CHECK(error);
        }

        return device_vector.num;
    }
};


template<typename T>
struct DOUBLE_ARRAY
{
    using TCUDAPtr = T*;
    using THOSTPtr = T*;

    THOSTPtr* HOST_mem;
    TCUDAPtr* CUDA_mem;

    size_t size;

    DOUBLE_ARRAY(size_t num, bool zeros = true)
    {
        size = sizeof(T) * num;

        HOST_mem = (T*)malloc(size);

        uint32_t error = cudaMalloc(&CUDA_mem, size);
        CUDA_CHECK(error);

        if (zeros)
        {
            memset(HOST_mem, 0, size);
            cudaMemset(CUDA_mem, 0, size);
        }
    }

    ~DOUBLE_ARRAY()
    {
        if (CUDA_mem)
        {
            cudaFree(CUDA_mem);
            CUDA_mem = nullptr;
        }

        if (HOST_mem)
        {
            free(HOST_mem);
            HOST_mem = nullptr;
        }
    }

    void write_GPU()
    {
        uint32_t error = cudaMemcpy(CUDA_mem, HOST_mem, size, cudaMemcpyHostToDevice);
        CUDA_CHECK(error);
    }

    void read_GPU()
    {
        uint32_t error = cudaMemcpy(HOST_mem, CUDA_mem, size, cudaMemcpyDeviceToHost);
        CUDA_CHECK(error);
    }
};


template<typename TTarget>
struct GpuOobject
{
    TTarget* CUDA_Ptr;
    TTarget* HOST_Ptr;

    GpuOobject(TTarget* source)
    {
        CUDA_Ptr = nullptr;
        HOST_Ptr = source;
    }

    ~GpuOobject()
    {
        if (CUDA_Ptr)
        {
            uint32_t error = cudaFree(CUDA_Ptr);
            CUDA_CHECK(error);
            CUDA_Ptr = nullptr;
        }
    }

    TTarget* ptr(bool sync = true)
    {
        if (CUDA_Ptr == nullptr)
        {
            uint32_t error = cudaMalloc(&CUDA_Ptr, sizeof(TTarget));
            CUDA_CHECK(error);
        }

        if (CUDA_Ptr && sync)
        {
            uint32_t error = cudaMemcpy(CUDA_Ptr, HOST_Ptr, sizeof(TTarget), cudaMemcpyHostToDevice);
            CUDA_CHECK(error);
        }

        return CUDA_Ptr;
    }

    void read()
    {
        if (CUDA_Ptr)
        {
            uint32_t error = cudaMemcpy(HOST_Ptr, CUDA_Ptr, sizeof(TTarget), cudaMemcpyDeviceToHost);
            CUDA_CHECK(error);
        }
    }
};

// Self owned GPU object
template<typename T>
struct TGenericGpuObject
{
    using TCudaPtr = T*;

    TGenericGpuObject(T* Self) : SelfGpu(Self) {
    }

    TCudaPtr ptr() {
        return SelfGpu.ptr();
    }

    void read() {
        SelfGpu.read();
    }

private:
    GpuOobject<T> SelfGpu;
};
