#pragma once

#include "common.h"

#include <vector>

#include "algorithm/keeloq/keeloq_learning_types.h"
#include "bruteforce/bruteforce_config.h"

// Forward declarations
typedef uint64_t EncData;

/**
 *  Aggregated configuration of application
 */
struct CommandLineArgs
{
    // Input encrypted data (3 caught OTA values)
    std::vector<EncData> inputs;

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
    // Checks if arguments enough for bruteforcing
    bool can_bruteforce();

    void init_inputs(const std::vector<EncData>& inp);

    void init_cuda(uint16_t b, uint16_t t, uint16_t l);

    // Check device capabilities and returns maximum thread allowed for single block
    static uint32_t max_cuda_threads();

    // Check device capabilities and returns maximum allowed number of blocks
    static uint32_t max_cuda_blocks();
};