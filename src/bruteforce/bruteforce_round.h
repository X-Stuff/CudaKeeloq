#pragma once

#include "common.h"

#include <vector>
#include <string>

#include "algorithm/keeloq/keeloq_learning_types.h"
#include "algorithm/keeloq/keeloq_single_result.h"
#include "algorithm/keeloq/keeloq_encrypted.h"
#include "algorithm/keeloq/keeloq_kernel_input.h"

#include "device/cuda_config.h"
#include "bruteforce/bruteforce_config.h"
#include "kernels/kernel_result.h"


/**
 *  Round is a set of bruteforce batches
 * Each batch runs `T` CUDA threads in `B` blocks
 * Each thread checks 1 or more decryptor
 * Each check is 1 or more (configured in args) keeloq learnings
 *
 * Typical is:
 *  Via command line some rounds were created - e.g. Dictionary and Simple attacks
 *  Each attack has N decryptors to check, through num blocks B, num threads T, and I iteration
 *  This means `(1 BATCH size) = B * T * I` decryptors check
 *  This means num of batches = `Total Num to Check / (1 BATCH size)`
 */
struct BruteforceRound
{
    // Construct round struct without specific learning type (means use all learnings)
    BruteforceRound(const std::vector<EncParcel>& data, const BruteforceConfig& gen, const CudaConfig& config) :
        BruteforceRound(data, gen, KeeloqLearning::Matrix::Everything(), config) {}

    // Construct round struct with only one selected learning type
    BruteforceRound(const std::vector<EncParcel>& data, const BruteforceConfig& gen, KeeloqLearning::Pair single, const CudaConfig& config) :
        BruteforceRound(data, gen, KeeloqLearning::Matrix({ single }), config) {}

    // Standard constructor
    BruteforceRound(const std::vector<EncParcel>& data, const BruteforceConfig& gen, const KeeloqLearning::Matrix& learning_matrix, const CudaConfig& config);

    ~BruteforceRound()
    {
        free();
    }

public:
    // Allocates memory
    void Init();

    // Reads results data from GPU memory into internal container and returns const reference to it
    bool read_results_gpu(std::vector<SingleResult>& container) const;

    // Checks Kernel's results
    // Return true if Round should be finished
    bool check_results(const KernelResult& result);

    // Get allocated memory amount for data
    size_t get_mem_size(bool cpu = false) const;

    // How many batches in this round (basically total keys to check divides by number of keys in a batch)
    size_t num_batches() const;

    // How many calculated results are in a batch (if use 3 inputs - 3 x keys_per_batch)
    size_t results_per_batch() const;

    // How many keys to check in this batch
    size_t keys_per_batch() const;

    std::string to_string() const;

public:
    inline const BruteforceConfig& Config() const { assert(inited); return kernel_inputs.GetConfig(); }

    inline BruteforceType::Type Type() const { assert(inited); return Config().type; }

    inline KeeloqKernelInput& Inputs() { assert(inited); return kernel_inputs; }

    inline const KeeloqKernelInput& Inputs() const { assert(inited); return kernel_inputs; }

private:

    void alloc();

    void free();

private:

    bool inited = false;

    // NumBlocks * NumThreads * [NumIterations] (num iteration is 1 if NO_INNER_LOOPS is defined, by default)
    uint32_t num_decryptors_per_batch = 0;

    //
    KeeloqKernelInput kernel_inputs;

    uint8_t encrypted_data_num = 0;

    CudaConfig cudaConfig;
};