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
#define LOG_ERROR(fmt, ...) APP_LOG_ERROR(verbosity, fmt, ##__VA_ARGS__)
#define LOG_PROGRESS(fmt, ...) APP_LOG_PROGRESS(verbosity, fmt, ##__VA_ARGS__)


Bruteforcer::Bruteforcer(const std::vector<EncParcel>& inputs, bool breakOnEsc, AppVerbosity verbosity) : inputs(inputs), breakOnEsc(breakOnEsc), verbosity(verbosity)
{

}

BruteforceResult Bruteforcer::run(const BruteforceConfig& config, const CudaConfig& cuda, const KeeloqLearning::Matrix& learningMatrix)
{
    auto hideCursor = console::ScopedHideCursor();

    BruteforceRound attackRound(inputs, config, cuda);

    LOG_INFO("\nAllocating...");
    attackRound.init();
    stats.allocatedBytesGPU = attackRound.getMemSize();
    LOG_INFO("\rRunning...");
    LOG_INFO("\n%s\n%s\n\n", attackRound.toString().c_str(), learningMatrix.toString(&attackRound.config()).c_str());

    return config.useSingleLearningKernels ?
        runSingle(attackRound, cuda, learningMatrix, config.getMutations()) :
        runMulti(attackRound, cuda, learningMatrix, config.getMutations());
}

BruteforceResult Bruteforcer::runMulti(BruteforceRound& round, const CudaConfig& cuda, const KeeloqLearning::Matrix& learningMatrix, const std::vector<InputsMutation> inputMutations)
{
    stats = Stats();

    if (round.isSingleLearningInputs())
    {
        LOG_ERROR("Round is NOT configured for multi-learning brute mode!");
        stats.result = Stats::InvalidConfiguration;
        return BruteforceResult::Invalid();
    }

    if (round.numBatches() == 0)
    {
        assert(false && "Invalid number of batches for attack round");
        LOG_ERROR("Invalid number of batches for attack round");
        stats.result = Stats::InvalidConfiguration;
        return BruteforceResult::Invalid();
    }

    // does nothing if already inited
    round.init();
    stats.allocatedBytesGPU = round.getMemSize();

    const size_t batchesInRound = round.numBatches();
    assert(batchesInRound > 0 && "Invalid number of batches for attack round");

    const size_t decryptorsInBatch = round.keysPerBatch();

    // Total checks per batch is keys in batch multiplied by number of input mutations
    stats.checksInBatch = decryptorsInBatch * inputMutations.size();

    auto stop = false;
    auto roundTimer = Timer<std::chrono::steady_clock>::start();
    auto batchTimer = Timer<std::chrono::steady_clock>::start();

    LOG_PROGRESS("\n\n");
    KernelResult lastResult;
    for (size_t batch = 0; !stop && batch < batchesInRound; ++batch)
    {
        if (!round.prepareInputs(batch))
        {
            stats.result = Stats::KernelFailure;
            return BruteforceResult::Invalid();
        }

        for (auto mIndex = 0; !stop && mIndex < inputMutations.size(); ++mIndex)
        {
            // Setup learning matrix and inputs mutation for this batch
            const auto currentInputMutation = inputMutations[mIndex];
            round.prepareBatch(learningMatrix, currentInputMutation);

            // do the bruteforce
            lastResult = keeloq::kernels::cuda_brute(round, cuda);
            stop = round.checkResults(lastResult, verbosity);

            printBruteforceProgress(round, batchTimer.elapsed().count(), roundTimer.elapsedSeconds(), batch, mIndex, inputMutations.size(), currentInputMutation);

            if (breakOnEsc && console::readEscPress())
            {
                LOG_PROGRESS("Bruteforce stopped by user\n");
                stats.result = Stats::UserCancelled;

                return BruteforceResult::Invalid();
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
        onRoundComplete(round, lastResult);
    }

    if (lastResult.hasMatch())
    {
        return getMatchResult(round);
    }

    if (stop && !lastResult.hasMatch())
    {
        // print was in `checkResults`
        return BruteforceResult::Invalid();
    }

    LOG_INFO("\n\nAfter: %zd batches no results was found. Keys checked:%zd\n\n", batchesInRound, stats.totalChecks());
    return BruteforceResult::Invalid();
}

BruteforceResult Bruteforcer::runSingle(BruteforceRound& round, const CudaConfig& cuda, const KeeloqLearning::Matrix& learningMatrix, const std::vector<InputsMutation> inputMutations)
{
    stats = Stats();

    if (!round.isSingleLearningInputs())
    {
        LOG_ERROR("Round is NOT configured for single learning brute mode!");
        stats.result = Stats::InvalidConfiguration;
        return BruteforceResult::Invalid();
    }

    if (round.numBatches() == 0)
    {
        assert(false && "Invalid number of batches for attack round");
        LOG_ERROR("Invalid number of batches for attack round");
        stats.result = Stats::InvalidConfiguration;
        return BruteforceResult::Invalid();
    }

    // does nothing if already inited
    round.init();
    stats.allocatedBytesGPU = round.getMemSize();

    const auto learningItems = learningMatrix.asItems();

    // Number of sub checks per batch
    const auto subChecksNum = learningItems.size() * inputMutations.size();

    // Total checks per batch is keys in batch multiplied by number of input mutations
    // Single mode does more checks but theoretically can do it faster due to greater number of allocated decryptors
    stats.checksInBatch = round.keysPerBatch() * subChecksNum;

    auto stop = false;
    auto roundTimer = Timer<std::chrono::steady_clock>::start();
    auto batchTimer = Timer<std::chrono::steady_clock>::start();

    LOG_PROGRESS("\n\n");
    KernelResult lastResult;
    for (size_t batch = 0; !stop && batch < round.numBatches(); ++batch)
    {
        if (!round.prepareInputs(batch))
        {
            stats.result = Stats::KernelFailure;
            return BruteforceResult::Invalid();
        }

        for (auto lIndex = 0; !stop && lIndex < learningItems.size(); ++lIndex)
        {
            const auto& learningItem = learningItems[lIndex];

            for (auto mIndex = 0; !stop && mIndex < inputMutations.size(); ++mIndex)
            {
                // Setup learning matrix (which is just single item) and inputs mutation for this batch
                const auto currentMutation = inputMutations[mIndex];
                round.prepareBatch(KeeloqLearning::Matrix { learningItem }, currentMutation);

                // do the bruteforce
                lastResult = keeloq::kernels::cuda_brute(round, cuda);
                stop = round.checkResults(lastResult, verbosity);

                const auto subCheckIndex = lIndex * inputMutations.size() + mIndex;
                printBruteforceProgress(round, batchTimer.elapsed().count(), roundTimer.elapsedSeconds(), batch, subCheckIndex, subChecksNum, currentMutation);

                if (breakOnEsc && console::readEscPress())
                {
                    LOG_PROGRESS("Bruteforce stopped by user\n");
                    stats.result = Stats::UserCancelled;

                    return BruteforceResult::Invalid();
                }
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
        onRoundComplete(round, lastResult);
    }

    if (lastResult.hasMatch())
    {
        return getMatchResult(round);
    }

    if (stop && !lastResult.hasMatch())
    {
        // print was in `checkResults`
        return BruteforceResult::Invalid();
    }

    LOG_INFO("\n\nAfter: %zd batches no results was found. Keys checked:%zd\n\n", round.numBatches(), stats.totalChecks());
    return BruteforceResult::Invalid();
}

void Bruteforcer::printBruteforceProgress(const BruteforceRound& round, const int64_t batchTime, const std::chrono::seconds& roundTime,
    const size_t batchIndex, const size_t subIndex, const size_t subNum, InputsMutation mutation)
{
    if (verbosity > AppVerbosity::Progress)
    {
        return;
    }

    const size_t batchesInRound = round.numBatches();
    const size_t keysInBatch = round.keysPerBatch();

    // Incremental, each mutation cycle will be increased
    const auto batchElapsedMs = batchTime;
    const auto calculatedNum = keysInBatch * (subIndex + 1);
    const auto avgPerBatchSpeed = batchElapsedMs / (subIndex + 1);

    const auto calculationIndex = (batchIndex * subNum + (subIndex + 1));
    const auto progressPercent = calculationIndex / static_cast<double>(batchesInRound * subNum);
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

BruteforceResult Bruteforcer::getMatchResult(const BruteforceRound& round, bool first)
{
    auto searchTimer = Timer<std::chrono::steady_clock>::start();
    return round.inputs().getMatch([&](auto curr, auto total)
        {
            printGpuMemorySearchProgress(curr, total, searchTimer.elapsedSeconds());
        });
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
