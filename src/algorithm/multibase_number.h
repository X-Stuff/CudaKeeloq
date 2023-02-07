#pragma once

#include "common.h"

#include <vector>
#include <cuda_runtime_api.h>


//
union U64Number
{
	uint64_t u64;

	uint8_t u8[8];
};


//
struct MultibaseNumber
{
	template<uint8_t TNum> friend struct MultibaseSystem;

	//
	__host__ __device__ uint64_t number() const { return value.u64; }

private:

	U64Number value   = {0};

	U64Number indices = {0};
};
