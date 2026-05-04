#pragma once

#include <cuda_runtime_api.h>

#include "common.h"


/**
 * Shared ownership over a host/GPU pair of `TTarget` instances.
 * `ptr()` lazily allocates and uploads; `read()` pulls device data back into the host copy.
 * The destructor frees the device allocation; the host pointer is caller-owned.
 */
template<typename TTarget>
struct CudaObject
{
	TTarget* CUDA_Ptr;
	TTarget* HOST_Ptr;

	CudaObject(TTarget* source)
	{
		CUDA_Ptr = nullptr;
		HOST_Ptr = source;
	}
	CudaObject(CudaObject<TTarget>&& other) = delete;

	CudaObject(const CudaObject<TTarget>& other) = delete;
	CudaObject<TTarget>& operator =(const CudaObject<TTarget>& other) = delete;

	~CudaObject()
	{
		if (CUDA_Ptr)
		{
			uint32_t error = cudaFree(CUDA_Ptr);
			CUDA_CHECK(error);
			CUDA_Ptr = nullptr;
		}

		HOST_Ptr = nullptr;
	}

	/** Returns the device pointer, allocating and (optionally) uploading from host on demand. */
	TTarget* ptr(bool sync = true)
	{
		if (HOST_Ptr == nullptr)
		{
			assert(false && "OBJECT IS NO LONGER CAN BE USED IN GPU");
			return nullptr;
		}

		if (CUDA_Ptr == nullptr)
		{
			auto error = cudaMalloc((void**)&CUDA_Ptr, sizeof(TTarget));
			CUDA_CHECK(error);
		}

		if (CUDA_Ptr && sync)
		{
			auto error = cudaMemcpy(CUDA_Ptr, HOST_Ptr, sizeof(TTarget), cudaMemcpyHostToDevice);
			CUDA_CHECK(error);
		}

		return CUDA_Ptr;
	}

	/** Pulls device contents back into the host copy (no-op before the first `ptr()`). */
	void read()
	{
		if (HOST_Ptr == nullptr)
		{
			assert(false && "OBJECT IS NO LONGER CAN BE USED IN GPU");
			return;
		}

		if (CUDA_Ptr)
		{
			auto error = cudaMemcpy(HOST_Ptr, CUDA_Ptr, sizeof(TTarget), cudaMemcpyDeviceToHost);
			CUDA_CHECK(error);
		}
	}
};

/**
 * CRTP mixin: a type derives from `TGenericGpuObject<Self>` to gain paired host/device storage
 * with `ptr()` / `read()` that operate on its own memory.
 */
template<typename T>
struct TGenericGpuObject
{
	using TCudaPtr = T*;

	TGenericGpuObject(T* Self) : SelfGpu(Self) {
	}

	TGenericGpuObject(const TGenericGpuObject<T>& other) = delete;
	TGenericGpuObject<T>& operator =(const TGenericGpuObject<T>& other) = delete;

	virtual TCudaPtr ptr()
	{
		return SelfGpu.ptr();
	}

	virtual void read()
	{
		SelfGpu.read();
	}

protected:

	CudaObject<T> SelfGpu;
};
