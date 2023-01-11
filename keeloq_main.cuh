#pragma once

#include <string>

#include "stdint.h"

#include "CUDA_helpers.cuh"

#define STRICT_ANALYSIS 1
#define NLF_LOOKUP_CONSTANT 0x3a5c742e


#define bit(x, n) (((x) >> (n)) & 1)
#define g5(x, a, b, c, d, e) \
    (bit(x, a) + bit(x, b) * 2 + bit(x, c) * 4 + bit(x, d) * 8 + bit(x, e) * 16)

// Input encoded data (received over the air) - 16bytes
typedef uint64_t EncData;

enum KeeloqLearningType
{
    Simple = 0,
    Simple_Rev,

    Normal,
    Normal_Rev,

    Secure,
    Secure_Rev,

    Xor,
    Xor_Rev,

    Faac,
    Faac_Rev,

    Serial1,
    Serial1_Rev,

    Serial2,
    Serial2_Rev,

    Serial3,
    Serial3_Rev,

    LAST,

    INVALID = 0xffffffff,
};

enum class GeneratorType : uint8_t
{
    // Generation will be skipped
    None = 0,
    Dictionary = 0,

    // Simple +1 generator
    Brute,

    // Excluding obvious patterns
    Smart,

    // Not for usage
    LAST,
};

static const char* LearningNames[KeeloqLearningType::LAST] = {
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

static const char* GeneratorTypeName[(int)GeneratorType::LAST] = {
    "Dictionary",
    "Brute",
    "Smart",
};

// fixed side array for every learning type
struct DecryptedArray
{
    uint32_t data[KeeloqLearningType::LAST];
};


struct SingleResult
{
    DecryptedArray results;

    uint64_t man;
    uint32_t seed;

    uint64_t ota;

    KeeloqLearningType match;
};


// is test manufacture code with seed (default is 0)
struct Decryptor
{
    uint64_t man;
    uint32_t seed;

    Decryptor() = default;
    Decryptor(uint64_t key, uint32_t s = 0) :
        man(key), seed(s) {}
};

struct DectyptorGenerationConfig
{
    // HOST SET. ONCE. How many generator rounds should be taken (in fact how many times cuda kernel will be called)
    size_t size;

    // HOST SET. ONCE. Decryption will start from this
    Decryptor start;

    // HOST SET. ONCE. Which generator to use.
    GeneratorType type;

    // GPU SET. UPDATING. Last generated decryptor (will be initial for next block run)
    Decryptor next;

    DectyptorGenerationConfig() :
        DectyptorGenerationConfig(GeneratorType::None, 0) {

    }

    DectyptorGenerationConfig(GeneratorType gen, size_t num) :
        DectyptorGenerationConfig(0, gen, num)
    {
        assert(gen == GeneratorType::Dictionary && "This constructor is available only for dictionary type");
    }

    DectyptorGenerationConfig(Decryptor start, GeneratorType gen, size_t num) :
        start(start), type(gen), next(start), size(num)
    {
    }

    uint64_t dict_size() const {
        if (type == GeneratorType::Dictionary) {
            return size;
        }
        return 0;
    }

    uint64_t brute_size() const {
        if (type != GeneratorType::Dictionary) {
            return size;
        }
        return 0;
    }

    void update_decryptor() {
        if (type != GeneratorType::Dictionary) {
            start = next;
        }
    }

    std::string ToString() const
    {
        char tmp[128];
        if (type == GeneratorType::Dictionary) {
            sprintf_s(tmp, "Type: %s. size: %zd", GeneratorTypeName[(uint8_t)type % (uint8_t)GeneratorType::LAST], dict_size());
        }
        else {
            sprintf_s(tmp, "Type: %s. Initial: 0x%llX (seed:%ul). Brute size: %zd",
                GeneratorTypeName[(uint8_t)type % (uint8_t)GeneratorType::LAST], start.man, start.seed, brute_size());
        }
        return std::string(tmp);
    }
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

    // from this dectryptor generation will start
    DectyptorGenerationConfig generator;

    KernelInput() : KernelInput(nullptr, nullptr, nullptr, DectyptorGenerationConfig())
    {
    }

    KernelInput(CUDA_Array<EncData>* enc, CUDA_Array<Decryptor>* dec, CUDA_Array<SingleResult>* res, const DectyptorGenerationConfig& config)
        : TGenericGpuObject<KernelInput>(this), encdata(enc), decryptors(dec), results(res), generator(config)
    {
    }

    KernelInput(KernelInput&& other) : TGenericGpuObject<KernelInput>(this) {
        encdata = other.encdata;
        decryptors = other.decryptors;
        results = other.results;
        generator = other.generator;
    }

    KernelInput& operator=(KernelInput&& other) = delete;
    KernelInput& operator=(const KernelInput& other) = delete;

    inline void WriteDecryptors(const std::vector<Decryptor>& source, size_t from, size_t num)
    {
        if (decryptors != nullptr)
        {
            generator.type = GeneratorType::Dictionary;

            size_t copy_num = max(0ull, min(num, (source.size() - from)));
            decryptors->write(&source[from], copy_num);
        }
    }

    inline void UpdateInitialDecryptor()
    {
        assert(generator.type != GeneratorType::Dictionary);
        generator.update_decryptor();
    }
};

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

    KernelResult(KernelResult&& other) : TGenericGpuObject<KernelResult>(this) {
        error = other.error;
        value = other.value;
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


// run decryption parallel per thread and find matches
__device__ uint8_t keeloq_decryption_run(const CUDACtx& ctx, CUDA_Array<EncData>* encrypted, CUDA_Array<Decryptor>* decryptors, CUDA_Array<SingleResult>* results);

// run from result[0] to result[num] tries to detect if there is a match (man key valid)
__device__ uint8_t keeloq_find_matches(const CUDACtx& ctx, SingleResult* results, uint32_t num);

// aggregate matches into count
__device__ uint8_t keeloq_analyze_results(const CUDACtx& ctx, const CUDA_Array<SingleResult>& results, uint32_t num_decryptors, uint32_t num_inputs);

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void CUDA_keeloq_test(KernelResult::TCudaPtr ret);

__global__ void CUDA_keeloq_main(KernelInput::TCudaPtr CUDA_inputs, KernelResult::TCudaPtr ret);

inline bool CUDA_check_keeloq_works()
{
    KernelResult kernel_results;
    CUDA_keeloq_test<<<1, 1>>>(kernel_results.ptr());
    kernel_results.read();

    return kernel_results.error == 0 && kernel_results.value != 0;
}


template<uint16_t ThreadBlocks, uint16_t ThreadsInBlock>
KernelResult CUDA_keeloq_main_wrapper(KernelInput& mainInputs, int&result, int& errors)
{
    KernelResult kernel_results;

    CUDA_keeloq_main<<<ThreadBlocks, ThreadsInBlock>>>(mainInputs.ptr(), kernel_results.ptr());

    mainInputs.read();
    kernel_results.read();

    return kernel_results;
}