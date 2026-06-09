#pragma once

#include <cstdlib>
#include <cstring>

#include <cuda_runtime_api.h>

#include "common.h"


/**
 * Paired host/device buffer; exposes explicit `writeGpu()` / `readGpu()` sync points.
 *
 * Ownership rules:
 *  - When constructed with a size, both host and device buffers are owned (and freed).
 *  - When constructed with an existing host pointer, the host buffer is borrowed.
 */
template<typename T>
struct DoubleArray
{
	using TCUDAPtr = T*;
	using THOSTPtr = T*;

	THOSTPtr HOST_mem;
	TCUDAPtr CUDA_mem;

	size_t size;

	DoubleArray(size_t num, bool zeros = true)
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

	DoubleArray(THOSTPtr array, size_t num)
		: hostOwner(false)
	{
		HOST_mem = array;
		size = sizeof(T) * num;

		cudaError error = cudaMalloc((void**) & CUDA_mem, size);
		CUDA_CHECK(error);

		error = cudaMemcpy(CUDA_mem, HOST_mem, size, cudaMemcpyHostToDevice);
		CUDA_CHECK(error);
	}

	~DoubleArray()
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

	/** Upload the host buffer's contents to the device buffer. */
	void writeGpu()
	{
		uint32_t error = cudaMemcpy(CUDA_mem, HOST_mem, size, cudaMemcpyHostToDevice);
		CUDA_CHECK(error);
	}

	/** Download the device buffer's contents into the host buffer. */
	void readGpu()
	{
		uint32_t error = cudaMemcpy(HOST_mem, CUDA_mem, size, cudaMemcpyDeviceToHost);
		CUDA_CHECK(error);
	}

private:
	bool hostOwner;
};
