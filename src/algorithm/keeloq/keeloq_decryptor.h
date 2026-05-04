#pragma once

#include <cuda_runtime_api.h>

#include "common.h"

#include "device/cuda_common.h"


/**
 * Decryption context holding the manufacturer key and (optionally) the seed.
 * Consumed directly by the CUDA kernels, so layout and member names are locked to device code.
 */
struct Decryptor
{
    __host__ __device__ Decryptor() : Decryptor(0)
    {
    }

    /** Creates a decryptor in an invalid (zero-key) state. */
    __host__ __device__ inline static Decryptor Invalid() { return Decryptor(); }

    /** Creates a decryptor from a key alone; seed is marked invalid. */
    __host__ __device__ inline static Decryptor MakeNoSeed(uint64_t k) { return Decryptor(k); }

    /**
     * Creates a decryptor with an explicit seed validity flag.
     * The explicit flag lets device code avoid branching — 0/-1 are legal seed values,
     * so they can't double as sentinels.
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
    /** Get the manufacturer key, optionally with reversed byte order (templated to avoid a runtime branch). */
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

    /** Whether this decryptor carries a meaningful seed (for seeded learning types). */
    __host__ __device__ __forceinline__ bool has_seed() const { return seed_valid; }

    /** Raw manufacturer key. */
    __host__ __device__ __forceinline__ uint64_t man() const { return key; }

    /** Seed for seeded learning types (undefined if has_seed() is false). */
    __host__ __device__ __forceinline__ uint32_t seed() const { return key_seed; }

    /** Manufacturer key with reversed byte order. */
    __host__ __device__ __forceinline__ uint64_t nam() const { return misc::rev_bytes(key); }

    /** Manufacturer key with reversed bit order. */
    __host__ __device__ __forceinline__ uint64_t nambits() const { return misc::rev_bits(key); }

    /** True if the decryptor was initialised with a non-zero key. */
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
