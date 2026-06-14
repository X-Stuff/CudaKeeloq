#pragma once

#include <chrono>
#include <functional>
#include <optional>
#include <vector>

#include "common.h"

#include "device/cuda_config.h"

#include "algorithm/keeloq/keeloq_decryptor.h"
#include "algorithm/keeloq/keeloq_encrypted.h"
#include "algorithm/keeloq/keeloq_learning_types.h"
#include "algorithm/keeloq/keeloq_thread_result.h"

#include "bruteforce/bruteforce_config.h"
#include "bruteforce/bruteforce_round.h"


using RoundCompleteCallback = std::function<void(const BruteforceRound& round, const KernelResult& result)>;


/**
 * Top-level driver that runs bruteforce rounds against a set of captured OTA inputs.
 * Each call to `run()` executes a full round described by a BruteforceConfig.
 */
struct Bruteforcer
{
    /**
     *  Special struct to capture various performance and outcome metrics for a single bruteforce round.
     */
    struct Stats
    {
        friend struct Bruteforcer;

        enum ResultType : uint8_t
        {
            NoMatch = 0,

            Success,

            InvalidConfiguration,

            KernelFailure,

            UserCancelled,
        };

    public:
        /** Whole round time */
        std::chrono::milliseconds roundTime = std::chrono::milliseconds(0);

        /**
         *  Number of total keys checked per batch (including input transforms) by single decryptor.
         */
        uint64_t keysInBatch = 0;

        /**
         *  Real number of total processed keys (all learnings, transforms, algorithm types), updated after every CUDA kernel launch.
         */
        uint64_t realProcessedKeys = 0;

        /** Average time in ms elapsed per batch (moving average) */
        double batchAverageMs = 0;

        /** Number of calculated batches */
        uint64_t numBatches = 0;

        /** Number of allocated byte on GPU */
        uint64_t allocatedBytesGPU = 0;

        ResultType result = NoMatch;
    public:
        /** Elapsed milliseconds for whole bruteforce round */
        inline uint64_t elapsedMs() const { return roundTime.count(); }

        /** Allocated GPU memory size in GB */
        inline float allocatedGB() const { return static_cast<float>(allocatedBytesGPU / (1024.0 * 1024.0 * 1024.0)); }

        /** Average round speed in thousands keys per second */
        inline double avgRoundSpeed() const { return static_cast<double>(realProcessedKeys / roundTime.count()); }

        /** Average batch speed in thousands keys per second */
        inline double avgBatchSpeed() const { return batchAverageMs != 0.0 ? keysInBatch / batchAverageMs : 0.0; }

    private:
        void setResult(const KernelResult& round, bool stopped);
    };

    /** Capture the OTA inputs that subsequent runs will attempt to decrypt. */
    Bruteforcer(const std::vector<EncParcel>& inputs, bool breakOnEsc = false, AppVerbosity verbosity = AppVerbosity::Debug);

public:
    /** Run one bruteforce round and return a matching result (or `BruteforceResult::Invalid()`). */
    BruteforceResult run(const BruteforceConfig& config, const CudaConfig& cuda);

    /** Set callback for round completion event*/
    void setOnRoundComplete(RoundCompleteCallback&& callback) { onRoundComplete = std::move(callback); }

    /** Returns bruteforce stats for the last run */
    const Stats& getStats() const { return stats; }

private:
    /**
     * Single unit of work within a batch: a learning matrix to apply and an input transform variant.
     * In multi-learning mode each entry carries the full matrix; in single-learning mode each entry
     * carries a one-item matrix so the kernel processes one learning type at a time.
     */
    struct SubCheck
    {
        /** Learning matrix to pass to the kernel for this sub-check */
        KeeloqLearning::Matrix matrix;

        /** Input transform to apply for this sub-check */
        InputsTransform transform;
    };

    struct ProgressInfo
    {
        struct BruteforceState
        {
            int64_t batchElapsedMs = 0;
            size_t keysInSubCheck = 0;
            size_t subIndex = 0;
            size_t subCount = 0;
            Decryptor lastDecryptor;
            InputsTransform transform = InputsTransform::None;
        };

        ProgressInfo(size_t stepIndex, size_t stepCount, std::chrono::seconds elapsed)
            : stepIndex(stepIndex), stepCount(stepCount), elapsed(elapsed) {}

        ProgressInfo(size_t stepIndex, size_t stepCount, std::chrono::seconds elapsed, BruteforceState bruteforce)
            : stepIndex(stepIndex), stepCount(stepCount), elapsed(elapsed), bruteforce(std::move(bruteforce)) {}

        size_t stepIndex;
        size_t stepCount;
        std::chrono::seconds elapsed;
        std::optional<BruteforceState> bruteforce;
    };

    BruteforceResult runImpl(BruteforceRound& round, const std::vector<SubCheck>& subChecks);

    BruteforceResult getMatchResult(const BruteforceRound& round, bool first = true);

    void printProgress(const ProgressInfo& info);

    // Returns valid inputs array, does some additional checks
    std::vector<EncParcel> makeInputs(const std::vector<EncParcel>& rawInputs);

private:
    // Input data for bruteforce (captured encoded)
    const std::vector<EncParcel> inputs;

    // Last round stats
    Stats stats;

    // Flag allows skip on Esc press (usefull in benchmark)
    bool breakOnEsc = false;

    // Flag enables silent mode (less output)
    AppVerbosity verbosity = AppVerbosity::Debug;

    // Optional callback to call on full round completion, allows access GPU memory before it will be freed. Useful for benchmarks and tests.
    RoundCompleteCallback onRoundComplete = nullptr;
};
