#pragma once

#include "common.h"

#include "device/cuda_object.h"


//
//
struct KernelResult : TGenericGpuObject<KernelResult>
{
	// num errors. negative are kernel errors. positive - number of threads error
	int error = 0;

	// overall result
	int value = 0;

	KernelResult() : TGenericGpuObject<KernelResult>(this) {
	}

	KernelResult(KernelResult&& other) noexcept : TGenericGpuObject<KernelResult>(this) {
		error = other.error;
		value = other.value;
	}

	KernelResult& operator=(KernelResult&& other) {
		error = other.error;
		value = other.value;
		SelfGpu.HOST_Ptr = this;
		return *this;
	}
};
