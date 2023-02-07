#pragma once

#include "common.h"

#include <cuda_runtime_api.h>

/**
 *  This is convenience wrapper allows for easier array management and
 * copying between CPU and GPU (and vice versa).
 *
 * WARNING:
 *	If this object is data owner (constructed without input data) - it will free memory in destructor
 *	If this object is just pointer container - will not do anything destructive with pointers in destructor
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