#pragma once

#include "common.h"

#include "algorithm/keeloq/keeloq_decryptor.h"
#include "algorithm/keeloq/keeloq_encrypted.h"
#include "algorithm/keeloq/keeloq_single_result.h"
#include "algorithm/keeloq/keeloq_learning_types.h"

#include "bruteforce/bruteforce_config.h"
#include "bruteforce/bruteforce_round.h"
#include "device/cuda_config.h"

#include <vector>

/**
 *  The high level hierarchy object
 */
struct Bruteforcer
{
    Bruteforcer(const std::vector<EncParcel>& inputs);

public:
    /**
     *  Run bruteforce with
     */
    SingleResult run(const BruteforceConfig& config, const CudaConfig& cuda, const KeeloqLearning::Matrix& learningMatrix);

private:
    SingleResult getMatchResult(const BruteforceRound& round, bool first = true);

private:
    // Input data for bruteforce (captured encoded)
    std::vector<EncParcel> inputs;
};