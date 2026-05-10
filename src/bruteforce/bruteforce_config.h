#pragma once

#include <string>
#include <vector>

#include <cuda_runtime_api.h>

#include "common.h"

#include "algorithm/keeloq/keeloq_decryptor.h"
#include "kernels/inputs_mutation.h"

#include "bruteforce/bruteforce_filters.h"
#include "bruteforce/bruteforce_pattern.h"
#include "bruteforce/bruteforce_type.h"


/**
 * Single-run attack descriptor: which generator to use, where to start,
 * and which decryptor/filter/pattern state is associated with it.
 */
struct BruteforceConfig
{
    static constexpr size_t MaxDecryptorsNum = static_cast<size_t>(-1);

public:

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

    BruteforceConfig() : BruteforceConfig(Decryptor::Invalid(), BruteforceType::LAST, InputsMutation::None, 0)
    {
    }

public:

    /** Dictionary attack over an explicit list of decryptors. */
    static BruteforceConfig GetDictionary(std::vector<Decryptor>&& dictionary, InputsMutation inputsMutation);

    /** Plain +1 bruteforce over a contiguous key range. */
    static BruteforceConfig GetBruteforce(Decryptor first, InputsMutation inputsMutation, size_t size);

    /** +1 bruteforce with include/exclude filters applied. */
    static BruteforceConfig GetBruteforce(Decryptor first, InputsMutation inputsMutation, size_t size, const BruteforceFilters& filters);

    /** +1 bruteforce over the 32-bit seed space for a fixed manufacturer key. */
    static BruteforceConfig GetSeedBruteforce(Decryptor first, InputsMutation inputsMutation, uint32_t size = static_cast<uint32_t>(-1));

    /** +1 bruteforce over the 32-bit space for a xor key for fixed part in ota. InputsMutation will be forced set to XorFix */
    static BruteforceConfig GetXorFixBruteforce(Decryptor first, InputsMutation inputsMutation, uint32_t size = static_cast<uint32_t>(-1));

    /** Alphabet (same byte set on every position) bruteforce. */
    static BruteforceConfig GetAlphabet(Decryptor first, InputsMutation inputsMutation, const MultibaseDigit& alphabet, size_t num = MaxDecryptorsNum, const std::string& name = "");

    /** Pattern bruteforce (per-position byte sets). */
    static BruteforceConfig GetPattern(Decryptor first, InputsMutation inputsMutation, const BruteforcePattern& pattern, size_t num = MaxDecryptorsNum);

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

public:
    /** override mutation mask, allow set it not as mask but as value - basically run only single mutation */
    void overrideMutationMask(InputsMutation mask, bool alone = false);

    /** Checks if specific inputs mutation is allowed in config. `None` is always allowed (if not XorFix bruteforce) */
    bool hasMutation(InputsMutation m) const;

    /** Get all allowed mutation as vector (useful for iteration) */
    std::vector<InputsMutation> getMutations() const;

    /** Human-readable description of the given mutation mask. */
    std::string mutationsToString() const;

private:
    BruteforceConfig(Decryptor start, BruteforceType::Type t, InputsMutation im, size_t num) :
        type(t), start(start), size(num), decryptors(), filters(), pattern(), last(start), allowedMutations(im)
    {
    }

private:
    // HOST SET. Additional mutation flags to expand into CPU-side kernel launch variants.
    // The unmutated input path is always included.
    InputsMutation allowedMutations = InputsMutation::None;

    // Flag allows force disable all other mutations except one
    bool maskAsSingleMutation = false;
};

inline std::vector<uint8_t> operator "" _b(const char* ascii, size_t num)
{
    return std::vector<uint8_t>(ascii, ascii + num);
}
