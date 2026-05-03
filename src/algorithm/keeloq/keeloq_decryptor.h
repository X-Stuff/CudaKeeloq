#pragma once

#include "common.h"

#include <cuda_runtime_api.h>

#include "device/cuda_common.h"


/**
 * Data struct which allows to decrypt encrypted data
 * In fact just (in most cases) just 64-bit integer (8 bytes array)
 */
struct Decryptor
{
    __host__ __device__ Decryptor() : Decryptor(0)
    {
    }

    /** Create without seed */
    __host__ __device__ inline static Decryptor Invalid() { return Decryptor(); }

    /** Create without seed */
    __host__ __device__ inline static Decryptor MakeNoSeed(uint64_t k) { return Decryptor(k); }

    /**
     *  Create with seed, and explicitly set seed validity
     * NOTE:
     *  We do use explicit validity flag since we do not want `if` blocks in CUDA code
     */
    __host__ __device__ inline static Decryptor Make(uint64_t k, uint32_t s, bool seed_valid) { return Decryptor(k, s, seed_valid); }

public:
	__host__ __device__ inline bool operator==(const Decryptor& other)
	{
		return key == other.key && key_seed == other.key_seed;
	}

	__host__ __device__ inline bool operator<(const Decryptor& other)
	{
		return key < other.key;
	}

public:
    // Get manufacturer key with reverse or normal byte order, depending on template parameter
    // Declared as template to optimize and prettyfy code
    template<bool IsReverse>
    __host__ __device__ __forceinline__ uint64_t getKey() const
    {
        if constexpr (IsReverse)
        {
            return nam();
        }
        else
        {
            return man();
        }
    }

    // If this decryptor has seed (for special learning types)
    __host__ __device__ __forceinline__ bool has_seed() const { return seed_valid; }

    // Get manufacturer key
    __host__ __device__ __forceinline__ uint64_t man() const { return key; }

    // Get seed
    __host__ __device__ __forceinline__ uint32_t seed() const { return key_seed; }

    // Get byte-reversed manufacturer key
    __host__ __device__ __forceinline__ uint64_t nam() const { return misc::rev_bytes(key); }

    // Get bit-reversed manufacturer key
    __host__ __device__ __forceinline__ uint64_t nambits() const { return misc::rev_bits(key); }

    // If decryptor was initialized properly
    __host__ __device__ __forceinline__ bool is_valid() const { return key != 0; }

private:
    __host__ __device__ Decryptor(uint64_t k) : Decryptor(k, 0, false) {}

    __host__ __device__ Decryptor(uint64_t k, uint32_t s, bool seed_valid) : key(k), key_seed(s), seed_valid(seed_valid) {}

protected:

    // manufacturer key
    uint64_t key = 0;

    // seed (for special learning types only)
    uint32_t key_seed = 0;

    // flag that shows that this decryptor's seed is valid
    // Since we cannot use 0 or -1, since they potentially can be valid seeds
    bool seed_valid = false;
};