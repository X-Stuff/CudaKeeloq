#pragma once

#include "common.h"

#include <vector>

#include "algorithm/keeloq/keeloq_learning_types.h"
#include "algorithm/keeloq/keeloq_encrypted.h"
#include "bruteforce/bruteforce_config.h"

#define APP_NAME "CudaKeeloq"

#define ARG_HELP "help"
#define ARG_TEST "test"
#define ARG_BENCHMARK "benchmark"
#define ARG_INPUTS "inputs"
#define ARG_BLOCKS "cuda-blocks"
#define ARG_THREADS "cuda-threads"
#define ARG_LOOPS "cuda-loops"
#define ARG_MODE "mode"
#define ARG_LTYPE "learning-type"
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

    // Do not do all 16 calculations, use predefined one
    std::vector<KeeloqLearningType::Type> selected_learning = {};

    //  Alphabets are just set of possible byte values
    // this sets may be shared between attacks
    std::vector<MultibaseDigit> alphabets;

    // Stop on first match
    bool match_stop;

    // Cuda setup
    uint16_t cuda_blocks;
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

    void init_cuda(uint16_t b, uint16_t t, uint16_t l);

    // Check device capabilities and returns maximum thread allowed for single block
    static uint32_t max_cuda_threads();

    // Check device capabilities and returns maximum allowed number of blocks
    static uint32_t max_cuda_blocks();
};