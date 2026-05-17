#include "bruteforce/bruteforcer.h"

#include <chrono>
#include <numeric>

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

BruteforceResult Bruteforcer::run(const BruteforceConfig& config, const CudaConfig& cuda, const KeeloqLearning::Matrix& fullMatrix)
{
    auto hideCursor = console::ScopedHideCursor();

    const auto reducedMatrix = config.reduceMatrix(fullMatrix);

    BruteforceRound attackRound(inputs, config, cuda);

    LOG_INFO("\nAllocating...");
    attackRound.init();
    stats.allocatedBytesGPU = attackRound.getMemSize();
    LOG_INFO("\rRunning...");
    LOG_INFO("\n%s\n%s\n\n", attackRound.toString().c_str(), reducedMatrix.toString().c_str());

    const auto inputMutations = config.getMutations();

    std::vector<SubCheck> subChecks;
    if (config.useSingleLearningKernels)
    {
        const auto learningItems = reducedMatrix.asItems();
        subChecks.reserve(learningItems.size() * inputMutations.size());

        for (const auto& item : learningItems)
        {
            if (KeeloqLearning::hasSeed(item.learning) && !config.hasSeed())
            {
                // If config doesn't have seed and learning requires seed - skip
                continue;
            }
            else if (config.type == BruteforceType::Seed)
            {
                // seedonly bruteforce type only meaningful for seeded learning types
                continue;
            }

            for (const auto& mutation : inputMutations)
            {
                subChecks.push_back({ KeeloqLearning::Matrix{ item }, mutation });
            }
        }
    }
    else
    {
        subChecks.reserve(inputMutations.size());
        for (const auto& mutation : inputMutations)
        {
            subChecks.push_back({ reducedMatrix, mutation });
        }
    }

    return runImpl(attackRound, subChecks);
}

BruteforceResult Bruteforcer::runImpl(BruteforceRound& round, const std::vector<SubCheck>& subChecks)
{
    stats = Stats();

    if (round.numBatches() == 0)
    {
        assert(false && "Invalid number of batches for attack round");
        LOG_ERROR("Invalid number of batches for attack round");
        stats.result = Stats::InvalidConfiguration;
        return BruteforceResult::Invalid();
    }

    round.init();
    stats.allocatedBytesGPU = round.getMemSize();

    const uint32_t numKeysInSubChecks = std::accumulate(subChecks.begin(), subChecks.end(), 0, [](uint32_t sum, const SubCheck& check)
        {
            return sum + check.matrix.numEnabled();
        });

    const size_t numBatches = round.numBatches();
    stats.keysInBatch = round.decryptorsPerBatch() * numKeysInSubChecks;

    auto stop = false;
    auto roundTimer = Timer<std::chrono::steady_clock>::start();
    auto batchTimer = Timer<std::chrono::steady_clock>::start();

    LOG_PROGRESS("\n\n");
    KernelResult lastResult;
    for (size_t batch = 0; !stop && batch < numBatches; ++batch)
    {
        if (!round.generateDecryptors(batch))
        {
            stats.result = Stats::KernelFailure;
            return BruteforceResult::Invalid();
        }

        for (size_t si = 0; !stop && si < subChecks.size(); ++si)
        {
            const auto& [matrix, mutation] = subChecks[si];

            lastResult = round.update(matrix, mutation);
            stop = round.checkResults(lastResult, verbosity);

            printBruteforceProgress(round, roundTimer.elapsedSeconds(), batch, batchTimer.elapsed().count(), si, subChecks.size(), matrix.numEnabled(), mutation);

            // Single subcheck process all decryptors multiplied by (1 or up to 11) keys depending on learning matrix
            stats.realProcessedKeys += round.decryptorsPerBatch() * matrix.numEnabled();

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
        return BruteforceResult::Invalid();
    }

    LOG_INFO("\n\nAfter: %zd batches no results was found. Keys checked:%zd\n\n", numBatches, stats.realProcessedKeys);
    return BruteforceResult::Invalid();
}

void Bruteforcer::printBruteforceProgress(const BruteforceRound& round, const std::chrono::seconds& roundTime,
    const size_t batchIndex, const int64_t batchTime, const size_t subIndex, const size_t subNum, const uint8_t learningsNum, InputsMutation mutation)
{
    if (verbosity > AppVerbosity::Progress)
    {
        return;
    }

    const size_t batchesInRound = round.numBatches();
    const size_t keysInBatch = round.decryptorsPerBatch();
    const size_t keysInSubCheck = keysInBatch * learningsNum;

    // Incremental, each mutation cycle will be increased
    const auto batchElapsedMs = batchTime;
    const auto calculatedNum = keysInSubCheck * (subIndex + 1);
    const auto avgPerBatchSpeed = (batchElapsedMs / (subIndex + 1)) * subNum;

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
