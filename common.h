#pragma once

#include "stdint.h"

#if __CUDA_ARCH__
	#define LOCATION
#else
	#ifdef CU_FILE
		#define LOCATION			// C++ compiler via NVCC - should be global NS also
	#else
		#define LOCATION CPU		// C++ compiler
	#endif
#endif


#ifdef CU_FILE
	/* .cu files should be all in global ns */

	#define NS_LOCATION_BEGIN
	#define NS_LOCATION_END

	#define NS_LOCATION
	#define USE_NS_LOCATION
#else
	#define NS_LOCATION namespace LOCATION
	#define USE_NS_LOCATION using namespace LOCATION;

	#define NS_LOCATION_BEGIN NS_LOCATION {
	#define NS_LOCATION_END }
#endif
