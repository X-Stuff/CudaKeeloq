#pragma once

#include "common.h"

#include <vector>

#include "algorithm/keeloq/keeloq_learning_types.h"
#include "algorithm/keeloq/keeloq_encrypted.h"
#include "bruteforce/bruteforce_config.h"

#define APP_NAME "CudaKeeloq"

#define ARG_HELP "help"
#define ARG_VERSION "version"
#define ARG_TEST "test"
#define ARG_BENCHMARK "benchmark"
#define ARG_INPUTS "inputs"
#define ARG_BLOCKS "cuda-blocks"
#define ARG_THREADS "cuda-threads"
#define ARG_LOOPS "cuda-loops"
#define ARG_MODE "mode"
#define ARG_LTYPE "learning-type"
#define ARG_CHECKREV "check-revkeys"
#define ARG_CHECKINV "check-invalgs"
#define ARG_WORDDICT "word-dict"
#define ARG_BINDICT "bin-dict"
#define ARG_BINDMODE "bin-dict-mode"
#define ARG_START "start"
#define ARG_SEED "seed"
#define ARG_COUNT "count"
#define ARG_ALPHABET "alphabet"
#define ARG_PATTERN "pattern"
#define ARG_IFILTER "include-filter"
#define ARG_EFILTER "exclude-filter"
#define ARG_FMATCH "first-match"


/**
 *  Aggregated configuration of application
 */
struct CommandLineArgs
{
    // Input encrypted data (3 caught OTA values)
    std::vector<EncParcel> inputs;

    // How brute will be performed (may be several iterations)
    std::vector<BruteforceConfig> brute_configs;

    // Do not do all 19 calculations, use predefined one
    std::vector<KeeloqLearning::LearningType> selected_learning = {};

    // Select specific modifications also
    KeeloqLearning::Modifier::Mask selected_mod_mask = KeeloqLearning::Modifier::Mask::Regular;

    //  Alphabets are just set of possible byte values
    // this sets may be shared between attacks
    std::vector<MultibaseDigit> alphabets;

    // Stop on first match
    bool match_stop;

    // Cuda setup
    uint32_t cuda_blocks;
    uint16_t cuda_threads;
    uint16_t cuda_loops;

    // run also tests
    bool run_tests;

    // Run only benchmarks (with selected values)
    bool run_bench;

public:

    // Parse from standard terminal way
    static CommandLineArgs parse(int argc, const char** argv);

public:
    // Checks if arguments enough for bruteforcing
    bool can_bruteforce();

    // Init enc parcel collection with raw OTA values
    void init_inputs(const std::vector<uint64_t>& inp);

    // Initialize CUDA setup with parameters, leave default values for the best setup
    void init_cuda(uint16_t blocks = 0, uint16_t threads = 0, uint8_t loops = 1);

    // Check device capabilities and returns maximum thread allowed for single block
    static uint32_t max_cuda_threads();

    // Check device capabilities and returns maximum allowed number of blocks
    static uint32_t max_cuda_blocks(uint8_t numSubSteps = 1);

    // Check device capabilities and returns maximum allowed number of blocks
    static size_t max_global_memory();
};