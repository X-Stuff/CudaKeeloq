#include "bruteforce/bruteforcer.h"

#include <chrono>

#include "algorithm/keeloq/keeloq_kernel.h"
#include "algorithm/keeloq/keeloq_learning_types.h"

#include "bruteforce/bruteforce_round.h"
#include "bruteforce/generators/generator_bruteforce.h"

#include "host/command_line_args.h"
#include "host/console.h"
#include "host/timer.h"

#define LOG_INFO(fmt, ...) APP_LOG_INFO(verbosity, fmt, ##__VA_ARGS__)
#define LOG_PROGRESS(fmt, ...) APP_LOG_PROGRESS(verbosity, fmt, ##__VA_ARGS__)


Bruteforcer::Bruteforcer(const std::vector<EncParcel>& inputs, bool breakOnEsc, AppVerbosity verbosity) : inputs(inputs), breakOnEsc(breakOnEsc), verbosity(verbosity)
{

}

SingleResult Bruteforcer::run(const BruteforceConfig& config, const CudaConfig& cuda, const KeeloqLearning::Matrix& learningMatrix)
{
    auto hideCursor = console::ScopedHideCursor();
    stats = Stats();

    BruteforceRound attackRound(inputs, config, cuda);
    const auto configInputMutations = config.getMutations();
    const auto numInputMutations = static_cast<uint8_t>(configInputMutations.size());

    LOG_INFO("\nAllocating...");
    attackRound.init();
    stats.allocatedBytesGPU = attackRound.getMemSize();
    LOG_INFO("\rRunning...");
    LOG_INFO("\n%s\n%s\n\n", attackRound.toString().c_str(), learningMatrix.toString(&config).c_str());

    KernelResult lastResult;

    const size_t batchesInRound = attackRound.numBatches();
    assert(batchesInRound > 0 && "Invalid number of batches for attack round");

    const size_t keysInBatch = attackRound.keysPerBatch();

    // Total calculations per batch is keys in batch multiplied by number of input mutations
    stats.batchCalcs = keysInBatch * numInputMutations;

    auto stop = false;
    auto roundTimer = Timer<std::chrono::steady_clock>::start();
    auto batchTimer = Timer<std::chrono::steady_clock>::start();

    LOG_PROGRESS("\n\n");
    for (size_t batch = 0; !stop && batch < batchesInRound; ++batch)
    {
        if (!attackRound.prepareInputs(batch))
        {
            stats.result = Stats::KernelFailure;
            return SingleResult::Invalid();
        }

        KeeloqKernelMultiLearningInput& kernelInput = attackRound.inputs();

        for (auto m = 0; !stop && m < numInputMutations; ++m)
        {
            // Setup learning matrix and inputs mutation for this batch
            kernelInput.BruteforcePrepare(learningMatrix, configInputMutations[m]);

            // do the bruteforce
            lastResult = keeloq::kernels::cuda_brute(kernelInput, cuda);
            stop = attackRound.checkResults(lastResult, verbosity);

            printBruteforceProgress(attackRound, batchTimer.elapsed().count(), roundTimer.elapsedSeconds(), batch, m, numInputMutations, configInputMutations[m]);

            if (breakOnEsc && console::readEscPress())
            {
                LOG_PROGRESS("Bruteforce stopped by user\n");
                stats.result = Stats::UserCancelled;

                return SingleResult::Invalid();
            }
        }

        const double batchDuration = batchTimer.reset().count() / 1.0;
        stats.numBatches++;
        stats.batchAverageMs += (batchDuration - stats.batchAverageMs) / stats.numBatches;
    }

    if (verbosity <= AppVerbosity::Progress)
    {
        console::clearLinesUp(2);
    }

    stats.roundTime = roundTimer.elapsed();
    stats.setResult(lastResult, stop);

    if (onRoundComplete)
    {
        onRoundComplete(attackRound, lastResult);
    }

    if (lastResult.hasMatch())
    {
        return getMatchResult(attackRound);
    }

    if (stop && !lastResult.hasMatch())
    {
        // print was in `checkResults`
        return SingleResult::Invalid();
    }

    LOG_INFO("\n\nAfter: %zd batches no results was found. Keys checked:%zd\n\n", batchesInRound, batchesInRound * keysInBatch * numInputMutations);
    return SingleResult::Invalid();
}

void Bruteforcer::printBruteforceProgress(const BruteforceRound& round, const int64_t batchTime, const std::chrono::seconds& roundTime,
    const size_t batchIndex, const uint8_t mutationIndex, const uint8_t mutationsNum, InputsMutation mutation)
{
    if (verbosity > AppVerbosity::Progress)
    {
        return;
    }

    const size_t batchesInRound = round.numBatches();
    const size_t keysInBatch = round.keysPerBatch();

    // Incremental, each mutation cycle will be increased
    const auto batchElapsedMs = batchTime;
    const auto calculatedNum = keysInBatch * (mutationIndex + 1);
    const auto avgPerBatchSpeed = batchElapsedMs / (mutationIndex + 1);

    const auto calculationIndex = (batchIndex * mutationsNum + (mutationIndex + 1));
    const auto progressPercent = calculationIndex / static_cast<double>(batchesInRound * mutationsNum);
    assert(progressPercent <= 1.0 && "Invalid percentage!");

    const Decryptor& lastUsedDecryptor = round.inputs().GetConfig().last;

    const double mResultPerSecond = batchElapsedMs == 0 ? 0 : calculatedNum / (batchElapsedMs * 1000.0);

    console_cursor_ret_up(2);

    // Overwrite lines
    printf("[%c][%zd/%zd]  %" PRIu64 "(ms)/batch, %4.1f Mk/s,  Last key:0x%" PRIX64 " (%u)  Last mutation: %-20s\n",
        WAIT_CHAR(calculationIndex), batchIndex, batchesInRound,
        avgPerBatchSpeed, mResultPerSecond, lastUsedDecryptor.man(), lastUsedDecryptor.seed(), name(mutation));
    console::progressBar(progressPercent, roundTime);
}

void Bruteforcer::printGpuMemorySearchProgress(const size_t index, const size_t count, const std::chrono::seconds& time)
{
    if (verbosity > AppVerbosity::Progress)
    {
        return;
    }

    // Progress bar
    console_cursor_ret_up(2);

    printf("[%c][%zd/%zd] Searching match in GPU memory...                                                                 \n",
        WAIT_CHAR(index), index, count);

    const auto progressPercent = (double)(index + 1) / count;
    console::progressBar(progressPercent, time);
}

SingleResult Bruteforcer::getMatchResult(const BruteforceRound& round, bool first)
{
    constexpr auto MaxElements = 1024 * 1024;

    const auto results = round.inputs().results->host();

    auto searchTimer = Timer<std::chrono::steady_clock>::start();

    const auto numIterations = results.num / MaxElements;

    for (size_t index = 0, count = 0; index < results.num; index += MaxElements, ++count)
    {
        printGpuMemorySearchProgress(count, numIterations, searchTimer.elapsedSeconds());

        // Read decryptors in batches, just to save RAM on host, using static method since we already copied object to host
        auto copied_results = CudaArray<SingleResult>::read(results, index, MaxElements);

        for (const auto& result : copied_results)
        {
            if (result.match != KeeloqLearning::NoMatch)
            {
                LOG_INFO("Found!\n");
                return result;
            }
        }
    }

    LOG_INFO("NOT FOUND!\n");
    return SingleResult::Invalid();
}

void Bruteforcer::Stats::setResult(const KernelResult& kResult, bool stopped)
{
    if (kResult.hasMatch())
    {
        result = Stats::Success;
    }
    else if (stopped)
    {
        result = Stats::KernelFailure;
    }
    else
    {
        result = Stats::NoMatch;
    }
}
