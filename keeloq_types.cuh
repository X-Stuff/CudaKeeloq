#pragma once

#include "common.h"

#include <string>
#include <algorithm>
#include <tuple>

#include "host/types/keeloq_learning_types.h"
#include "host/types/keeloq_single_result.h"
#include "host/types/keeloq_decryptor.h"

#include "host/types/bruteforce_filters.h"
#include "host/types/bruteforce_type.h"

#include "CUDA_helpers.cuh"

USE_NS_LOCATION

// Input encoded data (received over the air) - 16bytes
typedef uint64_t EncData;



// Single run Configuration
struct BruteforceConfig
{
    // CUDA bruteforce generator

    struct Alphabet
    {
        // Actual size of alphabet
        uint8_t num = 0;

        Alphabet() = default;

        // Duplicates will be removed
        Alphabet(const std::vector<uint8_t>& alphabet);

        __host__ __device__ inline uint64_t add(uint64_t number, uint64_t value) const
        {
            uint8_t result[8];
            *(uint64_t*)result = number;
            add(result, value);
            return *(uint64_t*)result;
        }

        __host__ __device__ inline void add(uint8_t number[sizeof(uint64_t)], uint64_t value) const
        {
#ifdef __CUDA_ARCH__
            #pragma unroll
#endif
            for (int i = 0; i < sizeof(uint64_t); ++i)
            {
                uint8_t digit = value % num;                // 17 % 6 = 5
                uint16_t addition = number[i] + digit;      // 5 + 5 = 10 (6 + 4)
                number[i] = addition % num;                 // n[i] = 4

                value /= num;                               // 17 / 6 = 2

                uint8_t carry = addition >= num;            // 10 > 6
                value += carry;                             // 2 + 1 = 3
            }
        }

        // return index of value (cannot fail - if value not in a LUT - always return 0 index)
        __host__ __device__ inline uint8_t lookup(uint8_t value) const {
            return lut[value];
        }

        // lookup foreach byte
        __host__ __device__ inline uint64_t lookup(uint64_t value) const
        {
            uint64_t result = 0;
            uint8_t* pResult = (uint8_t*)&result;
            uint8_t* pValue = (uint8_t*)&value;
#ifdef __CUDA_ARCH__
            #pragma unroll
#endif
            for (uint8_t i = 0; i < sizeof(uint64_t); ++i)
            {
                // Valid or 0 (first letter in alphabet)
                pResult[i] = lookup(pValue[i]);
            }
            return result;
        }

        // return value by index
        __host__ __device__ inline uint8_t operator[](uint8_t index) {
            return alp[index];
        }

        __host__ __device__ inline size_t invariants() const {
            return (size_t)pow(num, sizeof(uint64_t));
        }

        __host__ __device__ inline uint64_t value(uint64_t index) const {
            uint8_t* pIndex = (uint8_t*)&index;
            uint8_t result[8] = {
                alp[pIndex[0]],
                alp[pIndex[1]],
                alp[pIndex[2]],
                alp[pIndex[3]],
                alp[pIndex[4]],
                alp[pIndex[5]],
                alp[pIndex[6]],
                alp[pIndex[7]],
                };
            return *(uint64_t*)result;
        }

        __host__ std::string toString() const;

    private:
        static const uint16_t Size = 0xFF + 1; // 256

        // The alphabet itself (256 bytes max)
        uint8_t alp[Size] = {0};

        // The alphabet lookup table
        uint8_t lut[Size] = {0};

    };

    // HOST SET. ONCE. Which generator to use.
    BruteforceType::Type type;

    // HOST SET. UPDATING. PER BATCH. Decryption batch (or decrypters generation) will start from this
    Decryptor start;

    // HOST SET. ONCE. How many generator rounds should be taken (in fact how many times cuda kernel will be called)
    size_t size;

    // Dictionery - HOST SET. ONCE.
    // Brute -      GPU SET. UPDATING.
    std::vector<Decryptor> decryptors;

    // HOST SET. ONCE. for filtered type.
    BruteforceFilters filters;

    // HOST SET. ONCE. for alphabet type.
    Alphabet alphabet;

    // GPU SET. UPDATING. Last generated decryptor (will be initial for next block run)
    Decryptor next;

    BruteforceConfig() : BruteforceConfig(0, BruteforceType::LAST, 0) {
    }

    static BruteforceConfig GetDictionary(std::vector<Decryptor>&& dictionary)
    {
        BruteforceConfig result(0, BruteforceType::Dictionary, dictionary.size());
        result.decryptors = std::move(dictionary);
        return result;
    };

    static BruteforceConfig GetBruteforce(Decryptor first, size_t size) { return BruteforceConfig(first, BruteforceType::Simple, size); }

    static BruteforceConfig GetBruteforce(Decryptor first, size_t size, const BruteforceFilters& filters)
    {
        BruteforceConfig result(first, BruteforceType::Filtered, size);
        result.filters = filters;
        return result;
    }

    static BruteforceConfig GetAlphabet(Decryptor first, const Alphabet& alphabet, size_t num = (size_t)-1)
    {
        // max operation take to check all keys with alphabet of size
        num = std::min(alphabet.invariants(), num);

        BruteforceConfig result(first, BruteforceType::Alphabet, num);
        result.alphabet = alphabet;
        return result;
    }

    static BruteforceConfig GetPattern()
    {
        // not implemented
        return BruteforceConfig(0, BruteforceType::LAST, 0);
    }

    uint64_t dict_size() const {
        if (type == BruteforceType::Dictionary) {
            return size;
        }
        return 0;
    }

    uint64_t brute_size() const {
        if (type != BruteforceType::Dictionary) {
            return size;
        }
        return 0;
    }

    void update_decryptor() {
        if (type != BruteforceType::Dictionary) {
            start = next;
        }
    }

    std::string toString() const;

private:
    BruteforceConfig(Decryptor start, BruteforceType::Type t, size_t num) :
        start(start), type(t), next(start), size(num)
    {
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

    // Which type of learning use for decryption // the last one indicates all
    KeeloqLearningType::Type learning_types[KeeloqLearningType::LAST + 1];

    // from this dectryptor generation will start
    BruteforceConfig generator;

    KernelInput() : KernelInput(nullptr, nullptr, nullptr, BruteforceConfig())
    {
    }

    KernelInput(CUDA_Array<EncData>* enc, CUDA_Array<Decryptor>* dec, CUDA_Array<SingleResult>* res, const BruteforceConfig& config)
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
            assert(generator.type == BruteforceType::Dictionary);

            size_t copy_num = std::max(0ull, std::min(num, (source.size() - from)));
            decryptors->write(&source[from], copy_num);
        }
    }

    inline void UpdateInitialDecryptor()
    {
        assert(generator.type != BruteforceType::Dictionary);
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


inline std::vector<uint8_t> operator "" _b(const char* ascii, size_t num)
{
    return std::vector<uint8_t>(ascii, ascii + num);
}