#pragma once

#include <string>
#include <vector>

#include "common.h"

#include "algorithm/multibase_number.h"
#include "algorithm/multibase_system.h"


using PatternValue = MultibaseNumber;
using PatternSystem = MultibaseSystem<sizeof(uint64_t)>;

/**
 * Pattern / alphabet bruteforce description.
 *
 * Uses the multi-base "cylinder" abstraction:
 *  - Alphabet : 8 cylinders share one alphabet of bytes.
 *  - Pattern  : 8 cylinders each have their own set of bytes.
 */
struct BruteforcePattern
{
    /** Number of bytes (cylinders) in the pattern — matches the target key width. */
    static constexpr uint8_t BytesNumber = PatternSystem::DigitsNumber;

    BruteforcePattern() = default;

    /**
     * Construct from little-endian per-byte value sets (byte 0 is the lowest byte).
     * `pattern_string` is kept for debug printing.
     */
    BruteforcePattern(std::vector<std::vector<uint8_t>>&& pattern_bytes, const std::string& pattern_string = "");

    /**
     * Convenience constructor that uses the same alphabet for every byte.
     * Prefer BruteforceAlphabet at call sites — this overload exists for internal use.
     */
    BruteforcePattern(const MultibaseDigit& same_bytes, const std::string& name = "");

public:

    /** Initialize a pattern value from a raw 64-bit starting key. */
    __host__ __device__ inline PatternValue init(uint64_t begin) const;

    /** Advance `curr` by `amount` steps within the pattern. */
    __host__ __device__ inline PatternValue next(const PatternValue& curr, uint64_t amount) const;

    /** Total count of representable values in this pattern. */
    __host__ __device__ inline size_t size() const { return system.invariants(); }

    /** Per-byte numeral alphabet for a given byte index. */
    __host__ __device__ inline const MultibaseDigit& bytesVariants(uint8_t index) const;

public:

    /** Build a human-readable representation; `extended=true` enumerates every byte's alphabet. */
    __host__ std::string toString(bool extended = false) const;

public:
    /**
     * Parse the bytes accepted at a single position.
     * Accepts:
     *   0xDA       — single byte
     *   0x19-0x2A  — inclusive range
     *   *          — any byte
     *   0x91|0x23  — explicit set
     */
    static std::vector<uint8_t> parseBytes(std::string text);

    /** Try to parse a single byte written as 0xA1 or FF. */
    static bool tryParseSingleByte(std::string text, uint8_t& out);

    /** Try to parse a byte range written as 0x10-0x1F. */
    static std::vector<uint8_t> tryParseRangeBytes(std::string text);

protected:

    PatternSystem system;

    // **cannot and not designed** to be accessed on GPU
    std::string repr_string;
};

__host__ __device__ inline PatternValue BruteforcePattern::init(uint64_t begin) const
{
    return system.cast(begin);
}

__host__ __device__ inline PatternValue BruteforcePattern::next(const PatternValue& curr, uint64_t amount) const
{
    MultibaseNumber result = curr;
    return system.increment(result, amount);
}

__host__ __device__ inline const MultibaseDigit& BruteforcePattern::bytesVariants(uint8_t index) const
{
    assert(index < BytesNumber && "Invalid byte index, it's bigger that bytes count in this pattern");
    return system.getConfig(index);
}
