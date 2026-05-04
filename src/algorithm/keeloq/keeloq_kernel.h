#pragma once

#include <cuda_runtime_api.h>

#include "common.h"

#include "device/cuda_config.h"
#include "device/cuda_context.h"
#include "kernels/kernel_result.h"

#include "algorithm/keeloq/keeloq_kernel_input.h"
#include "algorithm/keeloq/keeloq_learning_types.h"

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



namespace keeloq::common
{
// This version like 5 times faster
__device__ __host__ inline uint32_t decrypt(const uint32_t data, const uint64_t key)
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

__device__ __host__ inline uint32_t encrypt(const uint32_t data, const uint64_t key) {
    uint32_t x = data, r;
    for (r = 0; r < 528; r++)
        x = (x >> 1) ^ ((bit(x, 0) ^ bit(x, 16) ^ (uint32_t)bit(key, r & 63) ^
            bit(NLF_LOOKUP_CONSTANT, g5(x, 1, 9, 20, 26, 31)))
            << 31);
    return x;
}

template<bool IsDecrypt = true>
__device__ __host__ inline uint32_t encdec(const uint32_t data, const uint64_t key)
{
    if constexpr (IsDecrypt)
    {
        return decrypt(data, key);
    }
    else
    {
        return encrypt(data, key);
    }
}
}

/**
 *  reference: https://github.com/DarkFlippers/unleashed-firmware/blob/dev/lib/subghz/protocols/keeloq_common.c
 * Getting real encryption key
 *
 * Additional variation of learning algorithms:
 *  - Using encrypt instead of decrypt for key generation (for some learning types)
 */
namespace keeloq::learning
{
    template<bool UseDecrypt = true>
    __device__ __host__ inline uint64_t secure(uint32_t data, uint32_t seed, const uint64_t key)
    {
        uint32_t k1, k2;

        data &= 0x0FFFFFFFUL;

        if constexpr (UseDecrypt)
        {
            k1 = keeloq::common::decrypt(data, key);
            k2 = keeloq::common::decrypt(seed, key);
        }
        else
        {
            k1 = keeloq::common::encrypt(data, key);
            k2 = keeloq::common::encrypt(seed, key);
        }

        return ((uint64_t)k1 << 32) | k2;
    }

    __device__ __host__ inline uint64_t magic_xor_type1(uint32_t data, uint64_t pxor)
    {
        data &= 0x0FFFFFFFUL;
        return (((uint64_t)data << 32) | data) ^ pxor;
    }

    template<bool UseDecrypt = true>
    __device__ __host__ inline uint64_t normal(uint32_t data, const uint64_t key)
    {
        uint32_t k1, k2;

        data &= 0x0FFFFFFFUL;
        if constexpr (UseDecrypt)
        {
            k1 = keeloq::common::decrypt(data | 0x20000000UL, key);
            k2 = keeloq::common::decrypt(data | 0x60000000UL, key);
        }
        else
        {
            k1 = keeloq::common::encrypt(data | 0x20000000UL, key);
            k2 = keeloq::common::encrypt(data | 0x60000000UL, key);
        }

        return ((uint64_t)k2 << 32) | k1; // key - shifrovanoya
    }

    template<bool UseDecrypt = false>
    __device__ __host__ inline uint64_t faac(const uint32_t seed, const uint64_t key)
    {
        const uint16_t hs = seed >> 16;
        const uint16_t ending = 0x544D;
        const uint32_t lsb = (uint32_t)hs << 16 | ending;

        if constexpr (UseDecrypt)
        {
            return (uint64_t)keeloq::common::decrypt(seed, key) << 32 | keeloq::common::decrypt(lsb, key);
        }
        else
        {
            return (uint64_t)keeloq::common::encrypt(seed, key) << 32 | keeloq::common::encrypt(lsb, key);
        }
    }

    __device__ __host__ inline uint64_t serial_type1(uint32_t data, uint64_t man)
    {
        return (man & 0xFFFFFFFF) | ((uint64_t)data << 40) |
            ((uint64_t)(((data & 0xff) + ((data >> 8) & 0xFF)) & 0xFF) << 32);
    }

    __device__ __host__ inline uint64_t serial_type2(uint32_t data, uint64_t man)
    {
        uint8_t* p = (uint8_t*)&data;
        uint8_t* m = (uint8_t*)&man;
        m[7] = p[0];
        m[6] = p[1];
        m[5] = p[2];
        m[4] = p[3];
        return man;
    }

    __device__ __host__ inline uint64_t serial_type3(uint32_t data, uint64_t man)
    {
        return (man & 0xFFFFFFFFFF000000) | (data & 0xFFFFFF);
    }
}


namespace keeloq
{
namespace kernels
{

// launch simple keeloq calculation on GPU to check if everything working
__host__ bool cuda_is_working();

// Main kernel launcher wrapper
__host__ KernelResult cuda_brute(KeeloqKernelInput& mainInputs, const CudaConfig& config);

// Single decrypt round with all learning types and modifications, used for testing and debugging
__host__ SingleResult cuda_encdec(uint64_t ota, uint64_t man, uint32_t seed, bool isDecrypt);

// Single decrypt round with all learning types and modifications, used for testing and debugging
__host__ __forceinline__ SingleResult cuda_enc(uint64_t ota, uint64_t man, uint32_t seed)
{
    return cuda_encdec(ota, man, seed, false);
}

// Single decrypt round with all learning types and modifications, used for testing and debugging
__host__ __forceinline__ SingleResult cuda_dec(uint64_t ota, uint64_t man, uint32_t seed)
{
    return cuda_encdec(ota, man, seed, true);
}

} // namespace kernels
} // namespace keeloq