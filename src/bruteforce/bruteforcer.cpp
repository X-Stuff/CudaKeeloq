#include "bruteforce/bruteforcer.h"

#include <chrono>

#include "algorithm/keeloq/keeloq_kernel.h"
#include "algorithm/keeloq/keeloq_learning_types.h"

#include "bruteforce/bruteforce_round.h"
#include "bruteforce/generators/generator_bruteforce.h"

#include "host/command_line_args.h"
#include "host/console.h"
#include "host/timer.h"


Bruteforcer::Bruteforcer(const std::vector<EncParcel>& inputs, bool breakOnEsc, bool silent) : inputs(inputs), breakOnEsc(breakOnEsc), silent(silent)
{

}

SingleResult Bruteforcer::run(const BruteforceConfig& config, const CudaConfig& cuda, const KeeloqLearning::Matrix& learningMatrix)
{
    stats = Stats();

    BruteforceRound attackRound(inputs, config, cuda);
    const auto configInputMutations = config.getMutations();
    const auto numInputMutations = configInputMutations.size();

    printf("\nAllocating...");
    attackRound.init();
    stats.allocatedBytesGPU = attackRound.getMemSize();
    printf("\rRunning...   ");

    if (!silent)
    {
        printf("\n%s\n%s\n\n", attackRound.toString().c_str(), learningMatrix.toString(&config).c_str());
    }

    KernelResult lastResult;

    const size_t batchesInRound = attackRound.numBatches();
    const size_t keysInBatch = attackRound.keysPerBatch();

    // Total calculations per batch is keys in batch multiplied by number of input mutations
    stats.batchCalcs = keysInBatch * numInputMutations;

    auto roundTimer = Timer<std::chrono::steady_clock>::start();
    auto stop = false;

    console::setCursorState(false);
    printf("\n\n");

    auto batchTimer = Timer<std::chrono::steady_clock>::start();

    for (size_t batch = 0; !stop && batch < batchesInRound; ++batch)
    {
        if (!attackRound.prepareInputs(batch))
        {
            stats.result = Stats::KernelFailure;
            return SingleResult::Invalid();
        }

        KeeloqKernelInput& kernelInput = attackRound.inputs();

        for (auto m = 0; !stop && m < numInputMutations; ++m)
        {
            // Setup learning matrix and inputs mutation for this batch
            kernelInput.BruteforcePrepare(learningMatrix, configInputMutations[m]);

            // do the bruteforce
            lastResult = keeloq::kernels::cuda_brute(kernelInput, cuda);
            stop = attackRound.checkResults(lastResult);

            {
                // Incremental, each mutation cycle will be increased
                const auto batchElapsed = batchTimer.elapsed().count();
                const auto calculatedNum = keysInBatch * (m + 1);
                const auto avgPerBatchSpeed = batchElapsed / (m + 1);

                const auto kResultPerSecond = batchElapsed == 0 ? 0 : calculatedNum / batchElapsed;
                const auto calculationIndex = (batch * numInputMutations + (m + 1));
                const auto progressPercent = calculationIndex / static_cast<double>(batchesInRound * numInputMutations);
                assert(progressPercent <= 1.0 && "Invalid percentage!");

                const Decryptor& lastUsedDecryptor = kernelInput.GetConfig().last;

                console_cursor_ret_up(2);

                // Overwrite lines
                printf("[%c][%zd/%zd]    %" PRIu64 "(ms)/batch, Speed: %" PRIu64 " KKeys/s   Last key:0x%" PRIX64 " (%u)  Last mutation: %s         \n",
                    WAIT_CHAR(calculationIndex), batch, batchesInRound, avgPerBatchSpeed, kResultPerSecond, lastUsedDecryptor.man(), lastUsedDecryptor.seed(), name(configInputMutations[m]));
                console::progressBar(progressPercent, roundTimer.elapsedSeconds());

                if (breakOnEsc && console::readEscPress())
                {
                    console::clearLinesUp(2);
                    printf("Bruteforce stopped by user\n");
                    stats.result = Stats::UserCancelled;

                    return SingleResult::Invalid();
                }
            }
        }

        const double batchDuration = batchTimer.reset().count() / 1.0;
        stats.numBatches++;
        stats.batchAverageMs += (batchDuration - stats.batchAverageMs) / stats.numBatches;

        if (stats.batchAverageMs > 1000000000)
        {
            int ib = 4;
            ib++;
        }
    }

    stats.roundTime = roundTimer.elapsed();

    console::clearLinesUp(2);

    if (lastResult.hasMatch())
    {
        stats.result = Stats::Success;
        return getMatchResult(attackRound);
    }

    if (stop && !lastResult.hasMatch())
    {
        // print was in `checkResults`
        stats.result = Stats::KernelFailure;
        return SingleResult::Invalid();
    }

    stats.result = Stats::NoMatch;
    if (!silent)
    {
        printf("\n\nAfter: %zd batches no results was found. Keys checked:%zd\n\n", batchesInRound, batchesInRound * keysInBatch * numInputMutations);
    }

    return SingleResult::Invalid();
}

SingleResult Bruteforcer::getMatchResult(const BruteforceRound& round, bool first)
{
    constexpr auto MaxElements = 1024 * 1024;

    const auto results = round.inputs().results->host();

    auto searchTimer = Timer<std::chrono::steady_clock>::start();

    const auto numIterations = results.num / MaxElements;

    for (size_t index = 0, count = 0; index < results.num; index += MaxElements, ++count)
    {
        // Progress bar
        console_cursor_ret_up(2);

        printf("[%c][%zd/%zd] Searching match in GPU memory...                                                                 \n",
            WAIT_CHAR(count), count, numIterations);

        const auto progressPercent = (double)(count + 1) / numIterations;
        console::progressBar(progressPercent, searchTimer.elapsedSeconds());


        // Read decryptors in batches, just to save RAM on host, using static method since we already copied object to host
        auto copied_results = CudaArray<SingleResult>::read(results, index, MaxElements);

        for (const auto& result : copied_results)
        {
            if (result.match != KeeloqLearning::NoMatch)
            {
                console::clearLinesUp(2);
                printf("Found!\n");
                return result;
            }
        }
    }

    console::clearLinesUp(2);
    printf("NOT FOUND!\n");
    return SingleResult::Invalid();
}