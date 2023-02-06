#pragma once

#include "common.h"

#include <cuda_runtime_api.h>


namespace misc
{
__host__ __device__ __forceinline__ uint64_t rev_bytes(uint64_t input)
{
#if __CUDA_ARCH__
    uint32_t key_lo = input;
    uint32_t key_hi = input >> 32;

    constexpr uint32_t selector_0 = 0x4567;
    constexpr uint32_t selector_1 = 0x0123;

    uint32_t key_rev_lo = __byte_perm(key_lo, key_hi, selector_0);
    uint32_t key_rev_hi = __byte_perm(key_lo, key_hi, selector_1);

    return ((uint64_t)key_rev_hi << 32) | key_rev_lo;
#else
    uint64_t input_rev = 0;
    uint64_t input_rev_byte = 0;
    for (uint8_t i = 0; i < 64; i += 8)
    {
        input_rev_byte = (uint8_t)(input >> i);
        input_rev = input_rev | input_rev_byte << (56 - i);
    }

    return input_rev;
#endif
}
}