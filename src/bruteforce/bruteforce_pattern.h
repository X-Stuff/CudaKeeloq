#pragma once

#include "common.h"

#include <vector>
#include <string>

#include "algorithm/multibase_system.h"
#include "algorithm/multibase_number.h"


using PatternValue = MultibaseNumber;
using PatternSystem = MultibaseSystem<sizeof(uint64_t)>;

/**
 *  In pattern mode "cylinders" are different sizes
 * Alphabet - 8 cylinders with same set of bytes
 * Pattern  - 8 cylinders with different set of bytes
 *
 */
struct BruteforcePattern
{
    // How many bytes in this pattern - this basically represent bye-length of the bruteforce target
    // if bruteforce target is 64-bit key - this number should be 8
    static constexpr uint8_t BytesNumber = PatternSystem::DigitsNumber;

    BruteforcePattern() = default;

    //  Create pattern from bytes.
    // LITTLE ENDIAN. pattern_bytes[0] is lowest byte in pattern. e.g. ( [ { 0x01 }, ... ] will ends as key 0x.....01 )
    // Pattern string is just for reference
    BruteforcePattern(std::vector<std::vector<uint8_t>>&& pattern_bytes, const std::string& pattern_string = "N/A");

    //  Special case constructor when patter is the same for every digit in underlying system
    // However this is not how this class supposed to be constructed by public
    // user-code should use @BruteforceAlphabet instead in that case
    BruteforcePattern(const MultibaseDigit& same_bytes);

public:

    // Initialize value for this pattern from 64-bit number
    __host__ __device__ inline PatternValue init(uint64_t begin) const;

    // Roll cylinders, increment @curr by @amount
    __host__ __device__ inline PatternValue next(const PatternValue& curr, uint64_t amount) const;

    // how many numbers are in this pattern
    __host__ __device__ inline size_t size() const { return system.invariants(); }

    // This pattern is fixed size (same as target attacking key size)
    // according to this pattern each byte can be only a specific value
    // with this method you can retrieve configuration of that byte
    __host__ __device__ inline const MultibaseDigit& bytes_variants(uint8_t index) const;

public:

    __host__ std::string to_string(bool extended = false) const;

public:
    // Convert possible single-byte pattern string to set of bytes
    // 0xDA      -> single byte
    // 0x19-0x2A -> range
    // *         -> full
    // 0x91;0x23 -> set of specific bytes
    static std::vector<uint8_t> ParseBytes(std::string text);

    // tries to parse single byte value like 0xA1 or FF
    static bool TryParseSingleByte(std::string text, uint8_t& out);

    // tries to parse single byte value like 0xA1 or FF
    static std::vector<uint8_t> TryParseRangeBytes(std::string text);

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

__host__ __device__ inline const MultibaseDigit& BruteforcePattern::bytes_variants(uint8_t index) const
{
    assert(index < BytesNumber && "Invalid byte index, it's bigger that bytes count in this pattern");
    return system.get_config(index);
}
