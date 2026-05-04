#pragma once

#include <vector>

#include "common.h"

#include "device/cuda_config.h"

#include "algorithm/keeloq/keeloq_decryptor.h"
#include "algorithm/keeloq/keeloq_encrypted.h"
#include "algorithm/keeloq/keeloq_learning_types.h"
#include "algorithm/keeloq/keeloq_single_result.h"

#include "bruteforce/bruteforce_config.h"
#include "bruteforce/bruteforce_round.h"


/**
 * Top-level driver that runs bruteforce rounds against a set of captured OTA inputs.
 * Each call to `run()` executes a full round described by a BruteforceConfig.
 */
struct Bruteforcer
{
    /** Capture the OTA inputs that subsequent runs will attempt to decrypt. */
    Bruteforcer(const std::vector<EncParcel>& inputs);

public:
    /** Run one bruteforce round and return a matching result (or `SingleResult::Invalid()`). */
    SingleResult run(const BruteforceConfig& config, const CudaConfig& cuda, const KeeloqLearning::Matrix& learningMatrix);

private:
    SingleResult getMatchResult(const BruteforceRound& round, bool first = true);

private:
    // Input data for bruteforce (captured encoded)
    std::vector<EncParcel> inputs;
};
