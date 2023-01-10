#pragma once

#include "stdint.h"

#include "CUDA_helpers.cuh"


#define NLF_LOOKUP_CONSTANT 0x3a5c742e

#define bit(x, n) (((x) >> (n)) & 1)
#define g5(x, a, b, c, d, e) \
    (bit(x, a) + bit(x, b) * 2 + bit(x, c) * 4 + bit(x, d) * 8 + bit(x, e) * 16)

#define KEELOQ_LEARNING_SIMPLE 0u
#define KEELOQ_LEARNING_SIMPLE_REV KEELOQ_LEARNING_SIMPLE + 1u

#define KEELOQ_LEARNING_NORMAL KEELOQ_LEARNING_SIMPLE_REV + 1u
#define KEELOQ_LEARNING_NORMAL_REV KEELOQ_LEARNING_NORMAL + 1u

#define KEELOQ_LEARNING_SECURE KEELOQ_LEARNING_NORMAL_REV + 1u
#define KEELOQ_LEARNING_SECURE_REV KEELOQ_LEARNING_SECURE + 1u

#define KEELOQ_LEARNING_MAGIC_XOR_TYPE_1 KEELOQ_LEARNING_SECURE_REV + 1u
#define KEELOQ_LEARNING_MAGIC_XOR_TYPE_1_REV KEELOQ_LEARNING_MAGIC_XOR_TYPE_1 + 1u

#define KEELOQ_LEARNING_FAAC KEELOQ_LEARNING_MAGIC_XOR_TYPE_1_REV + 1u
#define KEELOQ_LEARNING_FAAC_REV KEELOQ_LEARNING_FAAC + 1u

#define KEELOQ_LEARNING_MAGIC_SERIAL_TYPE_1 KEELOQ_LEARNING_FAAC_REV + 1u
#define KEELOQ_LEARNING_MAGIC_SERIAL_TYPE_1_REV KEELOQ_LEARNING_MAGIC_SERIAL_TYPE_1 + 1u

#define KEELOQ_LEARNING_MAGIC_SERIAL_TYPE_2 KEELOQ_LEARNING_MAGIC_SERIAL_TYPE_1_REV + 1u
#define KEELOQ_LEARNING_MAGIC_SERIAL_TYPE_2_REV KEELOQ_LEARNING_MAGIC_SERIAL_TYPE_2 + 1u

#define KEELOQ_LEARNING_MAGIC_SERIAL_TYPE_3 KEELOQ_LEARNING_MAGIC_SERIAL_TYPE_2_REV + 1u
#define KEELOQ_LEARNING_MAGIC_SERIAL_TYPE_3_REV KEELOQ_LEARNING_MAGIC_SERIAL_TYPE_3 + 1u


#define KEELOQ_LEARNING_LAST KEELOQ_LEARNING_MAGIC_SERIAL_TYPE_3_REV + 1u
#define KEELOQ_LEARNING_INVALID 0xff

static const char* LearningNames[KEELOQ_LEARNING_LAST] = {
    "KEELOQ_LEARNING_SIMPLE",
    "KEELOQ_LEARNING_SIMPLE_REV",
    "KEELOQ_LEARNING_NORMAL",
    "KEELOQ_LEARNING_NORMAL_REV",
    "KEELOQ_LEARNING_SECURE",
    "KEELOQ_LEARNING_SECURE_REV",
    "KEELOQ_LEARNING_MAGIC_XOR_TYPE_1",
    "KEELOQ_LEARNING_MAGIC_XOR_TYPE_1_REV",
    "KEELOQ_LEARNING_FAAC",
    "KEELOQ_LEARNING_FAAC_REV",
    "KEELOQ_LEARNING_MAGIC_SERIAL_TYPE_1",
    "KEELOQ_LEARNING_MAGIC_SERIAL_TYPE_1_REV",
    "KEELOQ_LEARNING_MAGIC_SERIAL_TYPE_2",
    "KEELOQ_LEARNING_MAGIC_SERIAL_TYPE_2_REV",
    "KEELOQ_LEARNING_MAGIC_SERIAL_TYPE_3",
    "KEELOQ_LEARNING_MAGIC_SERIAL_TYPE_3_REV",
};


// fixed side array for every learning type
struct DecryptedArray
{
    uint32_t data[KEELOQ_LEARNING_LAST];
};


struct SingleResult
{
    DecryptedArray results;

    uint64_t man;
    uint32_t seed;

    uint64_t ota;

    uint8_t match;
};

// Input encoded data (received over the air) - 16bytes
typedef uint64_t EncData;

// now just index in results vector. negative index - no match
typedef int32_t MatchData;

// is test manufacture code with seed (default is 0)
struct Decryptor
{
    uint64_t man;
    uint32_t seed;
};

enum class GeneratorType
{
    // Generation will be skipped
    None,

    // Simple +1 generator
    Brute,

    // Excluding obvious patterns
    Smart
};

struct DectyptorGenerationConfig
{
    Decryptor initial;

    GeneratorType type;
};


// Input data for main keeloq calculation kernel
struct KernelInput : TGenericGpuObject<KernelInput>
{
    // Constant per-run input data (captured encoded)
    CUDA_Array<EncData>* encdata;

    // Single-run set of dectryptors
    CUDA_Array<Decryptor>* decryptors;

    // Single-run results
    CUDA_Array<SingleResult>* results;
    CUDA_Array<MatchData>* matches;

    // from this dectryptor generation will start
    DectyptorGenerationConfig generation;

    KernelInput() : TGenericGpuObject<KernelInput>(this) {
    }

    void free()
    {
        if (encdata)
        {
            encdata->free();
            encdata = nullptr;
        }

        if (decryptors)
        {
            decryptors->free();
            decryptors = nullptr;
        }

        if (matches)
        {
            matches->free();
            matches = nullptr;
        }

        if (results)
        {
            results->free();
            results = nullptr;
        }
    }
};


//
//
struct KernelResult : TGenericGpuObject<KernelResult>
{
    using TCudaResultPtr = KernelResult*;

    // num errors. negative are kernel errors. positive - number of threads error
    int error = 0;

    // overall result
    int result = 0;

    KernelResult() : TGenericGpuObject<KernelResult>(this) {
    }
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ __host__ inline uint32_t keeloq_common_decrypt(const uint32_t data, const uint64_t key) {
    uint32_t x = data, r;
    for(r = 0; r < 528; r++)
        x = (x << 1) ^ bit(x, 31) ^ bit(x, 15) ^ (uint32_t)bit(key, (15 - r) & 63) ^
        bit(NLF_LOOKUP_CONSTANT, g5(x, 0, 8, 19, 25, 30));
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

__device__ __host__ struct DecryptedArray keeloq_decrypt_all(uint32_t data, uint32_t fix, const uint64_t key, const uint32_t seed);

__device__ __host__ struct DecryptedArray keeloq_decrypt(uint64_t ota, uint64_t man, uint32_t seed = 0);


__device__ uint8_t keeloq_find_matches(const CUDACtx& ctx, CUDA_Array<SingleResult>* results, uint32_t num_decryptors, uint32_t num_inputs);

__device__ uint8_t keeloq_analyze_results(const CUDACtx& ctx, const CUDA_Array<SingleResult>* results, CUDA_Array<MatchData>* matches);

__device__ void keeloq_decryption_run(const CUDACtx& ctx, CUDA_Array<EncData>* encrypted, CUDA_Array<Decryptor>* decryptors, CUDA_Array<SingleResult>* results);


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void CUDA_keeloq_main(KernelInput* CUDA_inputs, KernelResult::TCudaResultPtr ret);


template<uint16_t ThreadBlocks, uint16_t ThreadsInBlock>
void CUDA_keeloq_main_wrapper(KernelInput& mainInputs, int&result, int& errors)
{
    KernelResult kernel_results = KernelResult();

    CUDA_keeloq_main<<<ThreadBlocks, ThreadsInBlock>>>(mainInputs.ptr(), kernel_results.ptr());

    mainInputs.read();
    kernel_results.read();

    result = kernel_results.result;
    errors = kernel_results.error;
}