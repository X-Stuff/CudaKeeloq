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


#define CUDA_FOR_THREAD_ID(ctx, i, count) for(uint32_t i = ctx.thread_id; i < count; i += ctx.thread_max)

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

    __host__ __device__ inline T& operator[](size_t index) {
        assert(index < num && "Index is out of range in CUDA_Array");
        return CUDA_data[index];
    }
    __host__ __device__ inline const T& operator[](size_t index) const {
        assert(index < num && "Index is out of range in CUDA_Array");
        return CUDA_data[index];
    }

    inline void free()
    {
        CUDA_Array<T>::free(this);
    }

    inline void write(const T* source, size_t num)
    {
        CUDA_Array<T>::write(this, source, num);
    }

    inline size_t copy(std::vector<T>& target)
    {
        return CUDA_Array<T>::copy(this, target);
    }

    inline CUDA_Array<T> host()
    {
        // thiscall should work even with invalid pointer
        return CUDA_Array<T>::host(this);
    }

    // Copies this pointer (which assumes to be GPU) to host and return copy of the last element
    inline T host_last()
    {
        // thiscall should work even with invalid pointer
        CUDA_Array<T> HOST_array = CUDA_Array<T>::host(this);
        return CUDA_Array<T>::read(HOST_array, HOST_array.num - 1);
    }

    static CUDA_Array<T>* allocate(const std::vector<T>& source)
    {
        // Allocate memory of vector itself (ptr + size_t == 16 bytes)
        CUDA_Array<T>* result = nullptr;
        uint32_t error = cudaMalloc(&result, sizeof(CUDA_Array<T>));
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

    static CUDA_Array<T> host(const CUDA_Array<T>* device)
    {
        assert(device && "Invalid CUDA pointer");
        CUDA_Array<T> HOST_dest = {0};

        auto error = cudaMemcpy(&HOST_dest, device, sizeof(CUDA_Array<T>), cudaMemcpyDeviceToHost);
        CUDA_CHECK(error);

        return HOST_dest;
    }

    static void free(CUDA_Array<T>* array)
    {
        CUDA_Array<T> HOST_array = host(array);

        if (HOST_array.CUDA_data)
        {
            auto error = cudaFree(HOST_array.CUDA_data);
            CUDA_CHECK(error);
        }

        auto error = cudaFree(array);
        CUDA_CHECK(error);
    }

    static T read(const CUDA_Array<T>& HOST_Array, size_t index)
    {
        assert(index < HOST_Array.num);

        T result;
        auto error = cudaMemcpy(&result, &HOST_Array.CUDA_data[index], sizeof(T), cudaMemcpyDeviceToHost);
        CUDA_CHECK(error);

        return result;
    }

    static size_t copy(const CUDA_Array<T>* array, std::vector<T>& target)
    {
        CUDA_Array<T> HOST_array = host(array);
        if (HOST_array.num > 0)
        {
            size_t allocated_bytes = HOST_array.num * sizeof(T);
            target.reserve(HOST_array.num);

            auto error = cudaMemcpy((T*)target.data(), HOST_array.CUDA_data, allocated_bytes, cudaMemcpyDeviceToHost);
            CUDA_CHECK(error);
        }

        return HOST_array.num;
    }

    static void write(CUDA_Array<T>* dest, const T* source, size_t num)
    {
        CUDA_Array<T> HOST_dest = host(dest);

        assert(HOST_dest.CUDA_data && "CUDA Data wasn't allocated");
        if (HOST_dest.CUDA_data)
        {
            auto copy_bytes = sizeof(T) * min(num, HOST_dest.num);

            auto error = cudaMemcpy(HOST_dest.CUDA_data, source, copy_bytes, cudaMemcpyHostToDevice);
            CUDA_CHECK(error);
        }
    }
};


template<typename T>
struct DOUBLE_ARRAY
{
    using TCUDAPtr = T*;
    using THOSTPtr = T*;

    THOSTPtr HOST_mem;
    TCUDAPtr CUDA_mem;

    size_t size;

    DOUBLE_ARRAY(size_t num, bool zeros = true)
        : hostOwner(true)
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

    DOUBLE_ARRAY(THOSTPtr array, size_t num)
        : hostOwner(false)
    {
        HOST_mem = array;
        size = sizeof(T) * num;

        uint32_t error = cudaMalloc(&CUDA_mem, size);
        CUDA_CHECK(error);

        error = cudaMemcpy(CUDA_mem, HOST_mem, size, cudaMemcpyHostToDevice);
        CUDA_CHECK(error);
    }

    ~DOUBLE_ARRAY()
    {
        if (CUDA_mem)
        {
            cudaFree(CUDA_mem);
            CUDA_mem = nullptr;
        }

        if (HOST_mem && hostOwner)
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

private:
    bool hostOwner;
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
    GpuOobject(GpuOobject<TTarget>&& other) = delete;

    GpuOobject(const GpuOobject<TTarget>& other) = delete;
    GpuOobject<TTarget>& operator =(const GpuOobject<TTarget>& other) = delete;

    ~GpuOobject()
    {
        if (CUDA_Ptr)
        {
            uint32_t error = cudaFree(CUDA_Ptr);
            CUDA_CHECK(error);
            CUDA_Ptr = nullptr;
        }

        HOST_Ptr = nullptr;
    }

    TTarget* ptr(bool sync = true)
    {
        if (HOST_Ptr == nullptr)
        {
            assert(false && "OBJECT IS NO LONGER CAN BE USED IN GPU");
            return nullptr;
        }

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
        if (HOST_Ptr == nullptr)
        {
            assert(false && "OBJECT IS NO LONGER CAN BE USED IN GPU");
            return;
        }

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

    TGenericGpuObject(const TGenericGpuObject<T>& other) = delete;
    TGenericGpuObject<T>& operator =(const TGenericGpuObject<T>& other) = delete;

    virtual TCudaPtr ptr() {
        return SelfGpu.ptr();
    }

    void read() {
        SelfGpu.read();
    }

private:
    GpuOobject<T> SelfGpu;
};
