#pragma once

#include <vector>

#include "common.h"

#include "device/cuda_config.h"

#include "algorithm/keeloq/keeloq_encrypted.h"
#include "algorithm/keeloq/keeloq_learning_types.h"

#include "bruteforce/bruteforce_config.h"


#define APP_NAME "CudaKeeloq"

#define ARG_HELP "help"
#define ARG_VERSION "version"
#define ARG_BENCHMARK "benchmark"
#define ARG_INPUTS "inputs"
#define ARG_BLOCKS "cuda-blocks"
#define ARG_THREADS "cuda-threads"
#define ARG_LOOPS "cuda-loops"
#define ARG_MODE "mode"
#define ARG_LTYPE "learning-type"
#define ARG_NO_REGKEYS "no-reg-keys"
#define ARG_NO_NRMALGS "no-reg-algs"
#define ARG_CHECKREV "check-rev"
#define ARG_CHECKINV "check-inv"
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
 * Parsed command-line invocation — aggregates every option the CLI understands
 * and derives a CUDA launch configuration from it.
 */
struct CommandLineArgs
{
    // Input encrypted data (3 caught OTA values)
    std::vector<EncParcel> inputs;

    // How brute will be performed (may be several iterations)
    std::vector<BruteforceConfig> brute_configs;

    // Do not do all 19 calculations, use predefined one
    std::vector<KeeloqLearning::LearningType> selected_learning = {};

    // Select specific modifications for inputs
    std::vector<KeeloqLearning::Modifier::Input> selected_input_mods = {};

    // Select specific modifications for algorithm
    std::vector<KeeloqLearning::Modifier::Algo> selected_algo_mods = {};

    //  Alphabets are just set of possible byte values
    // this sets may be shared between attacks
    std::vector<MultibaseDigit> alphabets;

    // Stop on first match
    bool match_stop = false;

    // Run only benchmarks (with selected values)
    bool run_bench = false;

    // Run only benchmarks (with selected values)
    bool print_version = false;

public:
    /** Current CUDA launch config — auto-derived unless user set blocks/threads explicitly. */
    inline CudaConfig cudaConfig() const { return CudaConfig{ cuda_blocks, cuda_threads, cuda_loops }; }

public:

    /** Parse a standard argc/argv invocation into a CommandLineArgs. */
    static CommandLineArgs parse(int argc, const char** argv);

public:
    /** True if parsed arguments are sufficient to start a bruteforce run. */
    bool canBruteforce();

    /** Populate `inputs` from raw OTA values. */
    void initInputs(const std::vector<uint64_t>& inp);

    /** Configure CUDA launch parameters; 0 means "auto-pick the optimal value". */
    void initCuda(uint16_t blocks = 0, uint16_t threads = 0, uint8_t loops = 1);

private:
    // Cuda setup
    uint32_t cuda_blocks = 0;

    uint16_t cuda_threads = 0;

    uint16_t cuda_loops = 1;
};
