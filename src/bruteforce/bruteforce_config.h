#pragma once

#include "common.h"

#include <vector>
#include <string>

#include <cuda_runtime_api.h>

#include "algorithm/keeloq/keeloq_decryptor.h"

#include "bruteforce/bruteforce_pattern.h"
#include "bruteforce/bruteforce_filters.h"
#include "bruteforce/bruteforce_type.h"
#include "bruteforce/bruteforce_config.h"


/**
 *  Single run attack configuration
 * Run - selected type with specific parameters
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

    static BruteforceConfig GetDictionary(std::vector<Decryptor>&& dictionary);

    static BruteforceConfig GetBruteforce(Decryptor first, size_t size);

    static BruteforceConfig GetBruteforce(Decryptor first, size_t size, const BruteforceFilters& filters);

    static BruteforceConfig GetSeedBruteforce(Decryptor first, uint32_t size = static_cast<uint32_t>(-1));

    static BruteforceConfig GetAlphabet(Decryptor first, const MultibaseDigit& alphabet, size_t num = (size_t)-1);

    static BruteforceConfig GetPattern(Decryptor first, const BruteforcePattern& pattern, size_t num = (size_t)-1);

public:

    // Number of elements in dictionary (for dictionary type) or 0 for other types
    uint64_t dict_size() const;

    // Number of decryptors to generate (for non-dictionary types) or 0 for dictionary type
    uint64_t brute_size() const;

    // Update start and last decryptors to next values.
    // New start will be current last, and new last will be calculated according to generator type and size.
    void next_decryptor();

    // Returns true if config's start decryptor has seed, for regular bruteforce type
    // Returns true if config's type is seed bruteforce
    // Returns true if at least one of decryptors in dictionary has seed (for dictionary type)
    bool has_seed() const;

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