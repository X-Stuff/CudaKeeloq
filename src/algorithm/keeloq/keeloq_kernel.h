#pragma once

#include "common.h"

#include "device/cuda_context.h"
#include "kernels/kernel_result.h"

#include "algorithm/keeloq/keeloq_kernel_input.h"
#include "algorithm/keeloq/keeloq_learning_types.h"

#include <cuda_runtime_api.h>

#define NLF_LOOKUP_CONSTANT 0x3a5c742e

#ifdef NO_INNER_LOOPS
    #define KEELOQ_INNER_LOOP(ctx, index, num) uint32_t index = ctx.thread_id;
#else
    #define KEELOQ_INNER_LOOP(ctx, index, num) CUDA_FOR_THREAD_ID(ctx, index, num)
#endif // !NO_INNER_LOOPS



#define bit(x, n) (((x) >> (n)) & 1)
#define g5(x, a, b, c, d, e) \
    (bit(x, a) + bit(x, b) * 2 + bit(x, c) * 4 + bit(x, d) * 8 + bit(x, e) * 16)

// 0, 8, 19, 25, 30 == 0x42080101
#define g5dec(x, g) \
    auto m = (x & 0x42080101); \
    g = (0b11111 & ( m | (m >> 7) | (m >> 17) | ( m >> 22) | (m >> 26)))


__device__ __host__ inline uint32_t keeloq_common_decrypt_orig(const uint32_t data, const uint64_t key) {
    uint32_t x = data, r;
    for (r = 0; r < 528; r++)
        x = (x << 1) ^ bit(x, 31) ^ bit(x, 15) ^ (uint32_t)bit(key, (15 - r) & 63) ^
        bit(NLF_LOOKUP_CONSTANT, g5(x, 0, 8, 19, 25, 30));
    return x;
}

// This version like 5 times faster
__device__ __host__ inline uint32_t keeloq_common_decrypt(const uint32_t data, const uint64_t key)
{
    uint32_t x = data, g, k, f;
    int32_t r = 15;

    // outer 33 cycles
    for (uint8_t outer = 0; outer < 33; ++outer)
    {
        // Inner 16 cycles which could be unrolled (improves performance in release - decreases in debug)
        UNROLL
        for (uint8_t inner = 0; inner < 16; ++inner)
        {
            uint32_t key_bit = r & 0b111111;

            g5dec(x, g);

            k = (uint32_t)((key >> key_bit));
            f = ((x >> 31) ^ (x >> 15) ^ (NLF_LOOKUP_CONSTANT >> g) ^ k) & 1;
            x = (x << 1) ^ f;

            --r;
        }
    }


    return x;
}

__device__ __host__ inline uint32_t keeloq_common_encrypt(const uint32_t data, const uint64_t key) {
    uint32_t x = data, r;
    for (r = 0; r < 528; r++)
        x = (x >> 1) ^ ((bit(x, 0) ^ bit(x, 16) ^ (uint32_t)bit(key, r & 63) ^
            bit(NLF_LOOKUP_CONSTANT, g5(x, 1, 9, 20, 26, 31)))
            << 31);
    return x;
}

namespace keeloq
{
namespace kernels
{

// launch simple keeloq calculation on GPU to check if everything working
__host__ bool cuda_is_working();

// Main kernel launcher wrapper
__host__ KernelResult cuda_brute(KeeloqKernelInput & mainInputs, uint16_t ThreadBlocks, uint16_t ThreadsInBlock);

}

// Get enrcypted OTA data for specific configuration with key and learning ( xor simple and normal supported )
__host__ EncParcel GetOTA(uint64_t key, uint32_t seed, uint32_t serial, uint8_t button, uint16_t count, KeeloqLearningType::Type learning);

}