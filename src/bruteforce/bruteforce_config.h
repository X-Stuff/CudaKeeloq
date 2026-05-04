#pragma once

#include <string>
#include <vector>

#include <cuda_runtime_api.h>

#include "common.h"

#include "algorithm/keeloq/keeloq_decryptor.h"

#include "bruteforce/bruteforce_filters.h"
#include "bruteforce/bruteforce_pattern.h"
#include "bruteforce/bruteforce_type.h"


/**
 * Single-run attack descriptor: which generator to use, where to start,
 * and which decryptor/filter/pattern state is associated with it.
 */
struct BruteforceConfig
{
    // HOST SET. ONCE. Which generator to use.
    BruteforceType::Type type;

    // HOST SET. UPDATING. PER BATCH. Decryption batch (or decryptors generation) will start from this
    Decryptor start;

    // HOST SET. ONCE. How many generator rounds should be taken (in fact how many times CUDA kernel will be called)
    size_t size;

    // Dictionary - HOST SET. ONCE.
    // Brute -      GPU SET. UPDATING.
    std::vector<Decryptor> decryptors;

    // HOST SET. ONCE. for filtered type.
    BruteforceFilters filters;

    // HOST SET. ONCE. for pattern or alphabet type. (alphabet is just special case of pattern)
    BruteforcePattern pattern;

    // GPU SET. UPDATING. Last generated decryptor (will be initial for next block run)
    Decryptor last;

public:

    BruteforceConfig() : BruteforceConfig(Decryptor::Invalid(), BruteforceType::LAST, 0)
    {
    }

public:

    /** Dictionary attack over an explicit list of decryptors. */
    static BruteforceConfig GetDictionary(std::vector<Decryptor>&& dictionary);

    /** Plain +1 bruteforce over a contiguous key range. */
    static BruteforceConfig GetBruteforce(Decryptor first, size_t size);

    /** +1 bruteforce with include/exclude filters applied. */
    static BruteforceConfig GetBruteforce(Decryptor first, size_t size, const BruteforceFilters& filters);

    /** +1 bruteforce over the 32-bit seed space for a fixed manufacturer key. */
    static BruteforceConfig GetSeedBruteforce(Decryptor first, uint32_t size = static_cast<uint32_t>(-1));

    /** Alphabet (same byte set on every position) bruteforce. */
    static BruteforceConfig GetAlphabet(Decryptor first, const MultibaseDigit& alphabet, size_t num = (size_t)-1);

    /** Pattern bruteforce (per-position byte sets). */
    static BruteforceConfig GetPattern(Decryptor first, const BruteforcePattern& pattern, size_t num = (size_t)-1);

public:

    /** Dictionary entry count for Dictionary mode, 0 otherwise. */
    uint64_t dictSize() const;

    /** Generation budget for non-Dictionary modes, 0 otherwise. */
    uint64_t bruteSize() const;

    /**
     * Advance `start` to the next batch boundary (using generator-specific arithmetic).
     * New `start` is the previous `last`; new `last` is computed after the next batch runs.
     */
    void nextDecryptor();

    /**
     * Whether the active configuration meaningfully carries a seed.
     *  - Simple/Filtered/Alphabet/Pattern: reflects `start.has_seed()`
     *  - Seed: always true
     *  - Dictionary: true if any dictionary entry has a seed
     */
    bool hasSeed() const;

    /** Human-readable one-line description of the configuration. */
    std::string toString() const;

private:
    BruteforceConfig(Decryptor start, BruteforceType::Type t, size_t num) :
        type(t), start(start), size(num), decryptors(), filters(), pattern(), last(start)
    {
    }
};

inline std::vector<uint8_t> operator "" _b(const char* ascii, size_t num)
{
    return std::vector<uint8_t>(ascii, ascii + num);
}
