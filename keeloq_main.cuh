#pragma once

#include "stdint.h"

#include "keeloq_types.cuh"
#include "CUDA_helpers.cuh"

#define STRICT_ANALYSIS 1
#define NLF_LOOKUP_CONSTANT 0x3a5c742e


#define bit(x, n) (((x) >> (n)) & 1)
#define g5(x, a, b, c, d, e) \
    (bit(x, a) + bit(x, b) * 2 + bit(x, c) * 4 + bit(x, d) * 8 + bit(x, e) * 16)

// 0, 8, 19, 25, 30 == 0x42080101
#define g5dec(x, g) \
    auto m = (x & 0x42080101); \
    g = (0b11111 & ( m | (m >> 7) | (m >> 17) | ( m >> 22) | (m >> 26)))


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ __host__ inline uint32_t keeloq_common_decrypt_orig(const uint32_t data, const uint64_t key) {
    uint32_t x = data, r;
    for(r = 0; r < 528; r++)
        x = (x << 1) ^ bit(x, 31) ^ bit(x, 15) ^ (uint32_t)bit(key, (15 - r) & 63) ^
        bit(NLF_LOOKUP_CONSTANT, g5(x, 0, 8, 19, 25, 30));
    return x;
}

__device__ __host__ inline uint32_t keeloq_common_encrypt_orig(const uint32_t data, const uint64_t key) {
    uint32_t x = data, r;
    for(r = 0; r < 528; r++)
        x = (x >> 1) ^ ((bit(x, 0) ^ bit(x, 16) ^ (uint32_t)bit(key, r & 63) ^
            bit(NLF_LOOKUP_CONSTANT, g5(x, 1, 9, 20, 26, 31)))
            << 31);
    return x;
}


__device__ __host__ inline uint32_t keeloq_common_decrypt(const uint32_t data, const uint64_t key) {
    uint32_t x = data, g, k, f;

#if __CUDA_ARCH__
    #pragma unroll
#endif
    for (int32_t r = 15; r >= -512; --r)
    {
        uint32_t key_bit = r & 0b111111;

        g5dec(x, g);

        k = (uint32_t)((key >> key_bit));
        f = ((x >> 31) ^ (x >> 15) ^ (NLF_LOOKUP_CONSTANT >> g) ^ k) & 1;
        x = (x << 1) ^ f;
    }
    return x;
}


__device__ __host__ inline uint32_t keeloq_common_encrypt(const uint32_t data, const uint64_t key) {
    uint32_t x = data, r;
    for(r = 0; r < 528; r++)
        x = (x >> 1) ^ ((bit(x, 0) ^ bit(x, 16) ^ (uint32_t)bit(key, r & 63) ^
            bit(NLF_LOOKUP_CONSTANT, g5(x, 1, 9, 20, 26, 31)))
            << 31);
    return x;
}

__device__ __host__ SingleResult::DecryptedArray keeloq_decrypt_all(uint32_t data, uint32_t fix, const uint64_t key, const uint32_t seed);

__device__ __host__ SingleResult::DecryptedArray keeloq_decrypt(uint64_t ota, uint64_t man, uint32_t seed = 0);


// run decryption parallel per thread and find matches
__device__ uint8_t keeloq_decryption_run(const CUDACtx& ctx, CUDA_Array<EncData>* encrypted, CUDA_Array<Decryptor>* decryptors, CUDA_Array<SingleResult>* results);

// run from result[0] to result[num] tries to detect if there is a match (man key valid)
__device__ uint8_t keeloq_find_matches(const CUDACtx& ctx, SingleResult* results, uint32_t num);

// aggregate matches into count
__device__ uint8_t keeloq_analyze_results(const CUDACtx& ctx, const CUDA_Array<SingleResult>& results, uint32_t num_decryptors, uint32_t num_inputs);

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void CUDA_keeloq_test(KernelResult::TCudaPtr ret);

__global__ void CUDA_keeloq_main(KernelInput::TCudaPtr CUDA_inputs, KernelResult::TCudaPtr ret);

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

KernelResult CUDA_keeloq_main_wrapper(KernelInput& mainInputs, uint16_t ThreadBlocks, uint16_t ThreadsInBlock);


inline bool CUDA_check_keeloq_works()
{
    KernelResult kernel_results;
    CUDA_keeloq_test<<<1, 1>>>(kernel_results.ptr());
    kernel_results.read();

    return kernel_results.error == 0 && kernel_results.value != 0;
}

