#pragma once

#if __CUDA_ARCH__
	#define UNROLL #pragma unroll
#else
	#define UNROLL
#endif