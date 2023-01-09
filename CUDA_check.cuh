#pragma once

#include <assert.h>

#define CUDA_CHECK(error) \
    if (error != 0) { printf("ASSERTIN FAILED. CUDA ERROR!\n%s: %s\n", cudaGetErrorName((cudaError_t)error), cudaGetErrorString((cudaError_t)error)); }\
    assert(error == 0)
