#pragma once

#include <cstdio>
#include <vector>

#include "common.h"

#include "device/cuda_config.h"

#include "algorithm/keeloq/keeloq_encrypted.h"
#include "algorithm/keeloq/keeloq_learning_types.h"

#include "kernels/input_transform.h"

#include "bruteforce/bruteforce_config.h"


#define APP_NAME "CudaKeeloq"

#define ARG_HELP "help"
#define ARG_VERSION "version"
#define ARG_BENCHMARK "benchmark"
#define ARG_INPUTS "inputs"
#define ARG_BLOCKS "cuda-blocks"
#define ARG_THREADS "cuda-threads"
#define ARG_MODE "mode"
#define ARG_LTYPE "learning-type"
#define ARG_NO_REG_ALGS "no-reg-algs"
#define ARG_CHECK_INV_ALGS "check-inv-algs"
#define ARG_CHECK_REVKEYS "check-rev"
#define ARG_CHECK_XORFIX "check-xorfix"
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

    // Additional input transform flags selected by the user via CLI.
    InputsTransform inputsTransform = InputsTransform::None;

    //  Alphabets are just set of possible byte values
    // this sets may be shared between attacks
    std::vector<MultibaseDigit> alphabets;

    // Stop on first match
    bool match_stop = false;

    // Run only benchmarks (with selected values)
    bool run_bench = false;

    // Run only benchmarks (with selected values)
    bool print_version = false;

    // Set when the user requested help (-h, --help) or the invocation was
    // incomplete in a way that warrants showing usage. main() prints the help;
    // tests just check the flag.
    bool print_help = false;

    // Set when parsing encountered a fatal error (malformed/invalid argument).
    // Parsing stops at the first fatal error. main() exits with non-zero.
    bool has_errors = false;

public:
    inline uint32_t cudaBlocks() const { return cuda_blocks; }

    inline uint16_t cudaThreads() const { return cuda_threads; }

public:

    /** Parse a standard argc/argv invocation into a CommandLineArgs. */
    static CommandLineArgs parse(int argc, const char** argv, AppVerbosity verbosity = AppVerbosity::Debug);

    /** Print the full options help and usage examples. */
    static void printHelp(std::FILE* out = stdout);

public:
    /** True if parsed arguments are sufficient to start a bruteforce run. */
    bool canBruteforce();

    /** Populate `inputs` from raw OTA values. */
    void initInputs(const std::vector<uint64_t>& inp);

    /** Creates learning matrix from current command line arguments */
    KeeloqLearning::Matrix getLearningMatrix() const { return KeeloqLearning::Matrix(selected_learning, selected_algo_mods); }

private:
    // Brute all 7 or 11 learnings, or use predefined one
    std::vector<KeeloqLearning::LearningType> selected_learning = {};

    // Select specific modifications for algorithm
    std::vector<KeeloqLearning::Modifier::Algo> selected_algo_mods = {};


    // Cuda setup
    uint32_t cuda_blocks = 0;

    uint16_t cuda_threads = 0;
};
