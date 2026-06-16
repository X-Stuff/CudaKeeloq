#include "bruteforce/bruteforcer.h"

#include <algorithm>
#include <chrono>
#include <future>
#include <numeric>

#include "algorithm/keeloq/keeloq_kernel.h"
#include "algorithm/keeloq/keeloq_learning_types.h"

#include "bruteforce/bruteforce_round.h"
#include "bruteforce/generators/generator_bruteforce.h"

#include "host/command_line_args.h"
#include "host/console.h"
#include "host/timer.h"

#define LOG_INFO(fmt, ...) APP_LOG_INFO(verbosity, fmt, ##__VA_ARGS__)
#define LOG_WARNING(fmt, ...) APP_LOG_WARNING(verbosity, fmt, ##__VA_ARGS__)
#define LOG_ERROR(fmt, ...) APP_LOG_ERROR(verbosity, fmt, ##__VA_ARGS__)
#define LOG_PROGRESS(fmt, ...) APP_LOG_PROGRESS(verbosity, fmt, ##__VA_ARGS__)

/**
 *  Simple helper wrapper around std::future to keep code cleaner
 */
struct ScopedAwaiter
{
    ~ScopedAwaiter()
    {
        wait();
    }

    // Check if the future owner is still working
    bool busy() const
    {
        return future.valid() && future.wait_for(std::chrono::seconds(0)) != std::future_status::ready;
    }

    // Wait for the future to complete if it's valid
    void wait()
    {
        if (future.valid())
        {
            future.wait();
        }
    }

public:
    std::future<void> future;
};

Bruteforcer::Bruteforcer(const std::vector<EncParcel>& inInputs, bool breakOnEsc, AppVerbosity verbosity)
    : inputs(makeInputs(inInputs)), breakOnEsc(breakOnEsc), verbosity(verbosity)
{
}

BruteforceResult Bruteforcer::run(const BruteforceConfig& config, const CudaConfig& cuda)
{
    auto hideCursor = console::ScopedHideCursor();
    const auto& learningMatrix = config.getLearningMatrix();

    BruteforceRound attackRound(inputs, config, cuda);

    LOG_INFO("\nAllocating...");
    attackRound.init();
    stats.allocatedBytesGPU = attackRound.getMemSize();
    LOG_INFO("\rRunning...");
    LOG_INFO("\n%s\n%s\n\n", attackRound.toString().c_str(), learningMatrix.toString().c_str());

    const auto& inputTransfroms = config.getTransforms();

    std::vector<SubCheck> subChecks;
    if (config.useSingleLearningKernels)
    {
        const auto learningItems = learningMatrix.asItems();
        subChecks.reserve(learningItems.size() * inputTransfroms.size());

        for (const auto& item : learningItems)
        {
            for (const auto& transform : inputTransfroms)
            {
                subChecks.push_back({ KeeloqLearning::Matrix{ item }, transform });
            }
        }
    }
    else
    {
        subChecks.reserve(inputTransfroms.size());
        for (const auto& transform : inputTransfroms)
        {
            subChecks.push_back({ learningMatrix, transform });
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

    ScopedAwaiter printProgressAsync;

    for (size_t batch = 0; !stop && batch < numBatches; ++batch)
    {
        if (!round.generateDecryptors(batch))
        {
            stats.result = Stats::KernelFailure;
            return BruteforceResult::Invalid();
        }

        for (size_t si = 0; !stop && si < subChecks.size(); ++si)
        {
            const auto& [matrix, transform] = subChecks[si];

            lastResult = round.launch(matrix, transform);
            stop = round.checkResults(lastResult, verbosity);

            // Print progress async, save some milliseconds
            if (!printProgressAsync.busy())
            {
                ProgressInfo info(batch * subChecks.size() + si, numBatches * subChecks.size(), roundTimer.elapsedSeconds(),
                    ProgressInfo::BruteforceState
                    {
                        batchTimer.elapsed().count(),
                        round.decryptorsPerBatch() * matrix.numEnabled(),
                        si,
                        subChecks.size(),
                        round.inputs().GetConfig().last,
                        transform
                    });

                printProgressAsync.future = std::async(std::launch::async, &Bruteforcer::printProgress, this, std::move(info));
            }

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

    printProgressAsync.wait();

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

void Bruteforcer::printProgress(const ProgressInfo& info)
{
    if (verbosity > AppVerbosity::Progress)
    {
        return;
    }

    const auto progressPercent = static_cast<double>(info.stepIndex + 1) / info.stepCount;
    assert(progressPercent <= 1.0 && "Invalid percentage!");

    console_cursor_ret_up(2);
    std::fflush(stdout);

    if (info.bruteforce)
    {
        const auto& bf = *info.bruteforce;
        const auto calculatedNum = bf.keysInSubCheck * (bf.subIndex + 1);
        const auto avgPerBatchSpeed = (bf.batchElapsedMs / static_cast<int64_t>(bf.subIndex + 1)) * static_cast<int64_t>(bf.subCount);
        const double mResultPerSecond = bf.batchElapsedMs == 0 ? 0.0 : calculatedNum / (bf.batchElapsedMs * 1000.0);

        disable_word_wrap();
        printf("[%c][%zd/%zd]  %" PRId64 "(ms)/batch, %4.1f Mk/s, Last key:0x%" PRIX64 " (%u), TRS: %-33s\n",
            WAIT_CHAR(info.stepIndex), info.stepIndex, info.stepCount,
            avgPerBatchSpeed, mResultPerSecond, bf.lastDecryptor.man(), bf.lastDecryptor.seed(), InputTransformName(bf.transform).c_str());
        enable_word_wrap();
    }
    else
    {
        printf("[%c][%zd/%zd] Searching match in GPU memory...\n",
            WAIT_CHAR(info.stepIndex), info.stepIndex, info.stepCount);
    }

    console::progressBar(progressPercent, info.elapsed);
}


std::vector<EncParcel> Bruteforcer::makeInputs(const std::vector<EncParcel>& rawInputs)
{
    std::vector<EncParcel> inputs = rawInputs;
    assert(inputs.size() > 0);

    if (inputs.size() > 0)
    {
        const auto fix = inputs[0].fix();

        bool allSameFix = std::all_of(inputs.begin(), inputs.end(), [fix](const EncParcel& p) { return p.fix() == fix; });
        if (!allSameFix)
        {
            std::transform(inputs.begin(), inputs.end(), inputs.begin(), [](const EncParcel& p)
            {
                // Reversing, trying to check maybe misinputed format (hop|fix instead of fix|hop)
                return EncParcel(p.hop(), p.fix());
            });


            const auto newFix = inputs[0].fix();
            allSameFix = std::all_of(inputs.begin(), inputs.end(), [newFix](const EncParcel& p) { return p.fix() == newFix; });
            if (allSameFix)
            {
                LOG_WARNING("Your input data was reversed from (hop|fix) to (fix|hop)!");
            }
        }

        if (!allSameFix)
        {
            LOG_ERROR("Invalid inputs! All inputs must have the same fixed part (serial|button). Please check your input data.");
        }
    }

    if (inputs.size() < 3)
    {
        LOG_WARNING("Bruteforcer initialized with %zd inputs, but only 3 are supported. Duplicating first input to fill the rest.", inputs.size());

        // We do not support any other amount of inputs rather than3
        while (inputs.size() < 3)
        {
            inputs.push_back(inputs[0]);
        }
    }

    return inputs;
}


BruteforceResult Bruteforcer::getMatchResult(const BruteforceRound& round, bool first)
{
    auto searchTimer = Timer<std::chrono::steady_clock>::start();
    return round.inputs().getMatch([&](auto curr, auto total)
        {
            printProgress({ curr, total, searchTimer.elapsedSeconds() });
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
