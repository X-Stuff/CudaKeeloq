#pragma once

#include <string>
#include <vector>

#include <cuda_runtime_api.h>

#include "common.h"

#include "algorithm/keeloq/keeloq_decryptor.h"
#include "algorithm/keeloq/keeloq_learning_types.h"
#include "kernels/input_transform.h"

#include "device/cuda_config.h"

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
    std::vector<Decryptor> dictDecryptors;

    // HOST SET. ONCE. for filtered type.
    BruteforceFilters filters;

    // HOST SET. ONCE. for pattern or alphabet type. (alphabet is just special case of pattern)
    BruteforcePattern pattern;

    // GPU SET. UPDATING. Last generated decryptor (will be initial for next block run)
    Decryptor last;

    // Experimental, single learning kernel mode
    bool useSingleLearningKernels = false;

public:

    BruteforceConfig() : BruteforceConfig(Decryptor::Invalid(), BruteforceType::LAST, InputsTransform::None, 0)
    {
    }

public:

    /** Dictionary attack over an explicit list of decryptors. */
    static BruteforceConfig GetDictionary(std::vector<Decryptor>&& dictionary, InputsTransform inputTransform);

    /** Plain +1 bruteforce over a contiguous key range. */
    static BruteforceConfig GetBruteforce(Decryptor first, InputsTransform inTransform, size_t size);

    /** +1 bruteforce with include/exclude filters applied. */
    static BruteforceConfig GetBruteforce(Decryptor first, InputsTransform inTransform, size_t size, const BruteforceFilters& filters);

    /** +1 bruteforce over the 32-bit seed space for a fixed manufacturer key. */
    static BruteforceConfig GetSeedBruteforce(Decryptor first, InputsTransform inTransform, uint32_t size = static_cast<uint32_t>(-1));

    /** +1 bruteforce over the 32-bit space for a xor key for fixed part in ota. InputTransform will be forced set to XorFix */
    static BruteforceConfig GetXorFixBruteforce(Decryptor first, InputsTransform inTransform, uint32_t size = static_cast<uint32_t>(-1));

    /** Alphabet (same byte set on every position) bruteforce. */
    static BruteforceConfig GetAlphabet(Decryptor first, InputsTransform inTransform, const MultibaseDigit& alphabet, size_t num = MaxDecryptorsNum, const std::string& name = "");

    /** Pattern bruteforce (per-position byte sets). */
    static BruteforceConfig GetPattern(Decryptor first, InputsTransform inTransform, const BruteforcePattern& pattern, size_t num = MaxDecryptorsNum);

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

    /** Create the CUDA configuration for this bruteforce config. */
    CudaConfig cudaConfig(int desiredThreads, int desiredBlocks) const;

    /** Get the learning matrix for this bruteforce config. */
    const KeeloqLearning::Matrix& getLearningMatrix() const { return learningMatrix; }

    /** Human-readable one-line description of the configuration. */
    std::string toString() const;

public:
    /** Replace the transform schedule with an explicit list. */
    void setTransforms(std::vector<InputsTransform> schedule);

    /** Set the learning matrix for this bruteforce config, will be reduced according to config. */
    void setLearningMatrix(const KeeloqLearning::Matrix& matrix);

    /** Get the ordered list of input transforms to apply per batch. */
    const std::vector<InputsTransform>& getTransforms() const { return transforms; }

    /** Human-readable description of the configured transforms. */
    std::string transformsToString() const;

    /**
     *  Reduce some learning types depends on config,
     * e.g. Seed-Only bruteforce type doesn't require non-seed learning types,
     *  or if decryptors doesn't have seed, we should not brute learning types that require seed.
     */
    KeeloqLearning::Matrix reduceMatrix(const KeeloqLearning::Matrix& matrix) const;

private:
    BruteforceConfig(Decryptor start, BruteforceType::Type t, InputsTransform mask, size_t num);

private:
    // Ordered list of input transforms to try per batch (built from mask at construction).
    std::vector<InputsTransform> transforms;

    // Allowed keeloq learning type for this bruteforce
    KeeloqLearning::Matrix learningMatrix = KeeloqLearning::Matrix::Everything();
};

inline std::vector<uint8_t> operator "" _b(const char* ascii, size_t num)
{
    return std::vector<uint8_t>(ascii, ascii + num);
}
