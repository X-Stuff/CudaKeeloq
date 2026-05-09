#include "bruteforce/bruteforcer.h"

#include <chrono>

#include "algorithm/keeloq/keeloq_kernel.h"
#include "algorithm/keeloq/keeloq_learning_types.h"

#include "bruteforce/bruteforce_round.h"
#include "bruteforce/generators/generator_bruteforce.h"

#include "host/command_line_args.h"
#include "host/console.h"
#include "host/timer.h"


Bruteforcer::Bruteforcer(const std::vector<EncParcel>& inputs) : inputs(inputs)
{

}

SingleResult Bruteforcer::run(const BruteforceConfig& config, const CudaConfig& cuda, const KeeloqLearning::Matrix& learningMatrix)
{
    BruteforceRound attackRound(inputs, config, cuda);
    const auto configInputMutations = config.getMutations();
    const auto numInputMutations = configInputMutations.size();

    printf("\nAllocating...");
    attackRound.init();

    printf("\rRunning...    \n%s\n%s\n", attackRound.toString().c_str(), learningMatrix.toString(&config).c_str());

    KernelResult lastResult;

    const size_t batchesInRound = attackRound.numBatches();
    const size_t keysInBatch = attackRound.keysPerBatch();

    auto roundTimer = Timer<std::chrono::steady_clock>::start();
    auto stop = false;

    console::setCursorState(false);
    printf("\n\n\n");

    auto batchTimer = Timer<std::chrono::steady_clock>::start();

    for (size_t batch = 0; !stop && batch < batchesInRound; ++batch)
    {
        KeeloqKernelInput& kernelInput = attackRound.inputs();

        if (attackRound.type() != BruteforceType::Dictionary)
        {
            if (batch != 0)
            {
                // Make last decryptor from previous batch as first for this batch
                kernelInput.NextDecryptor();
            }

            // Generate decryptors (if available)
            auto cudaError = GeneratorBruteforce::PrepareDecryptors(kernelInput, cuda);
            if (cudaError != cudaSuccess)
            {
                printf("Error: Key generation resulted with error: %s: %s\n", cudaGetErrorName(cudaError), cudaGetErrorString(cudaError));
                assert(false);
                return SingleResult::Invalid();
            }
        }
        else
        {
            // Write next batch of keys from dictionary
            kernelInput.WriteDecryptors(config.decryptors, batch * keysInBatch, keysInBatch);
        }

        for (auto m = 0; !stop && m < numInputMutations; ++m)
        {
            // Setup learning matrix and inputs mutation for this batch
            kernelInput.BruteforcePrepare(learningMatrix, configInputMutations[m]);

            // do the bruteforce
            lastResult = keeloq::kernels::cuda_brute(kernelInput, cuda);
            stop = attackRound.checkResults(lastResult);

            {
                const auto bruteCallDuration = batchTimer.reset().count();

                const auto kResultPerSecond = bruteCallDuration == 0 ? 0 : keysInBatch / bruteCallDuration;
                const auto progressPercent = (double)(batch + 1) / batchesInRound;

                const Decryptor& lastUsedDecryptor = kernelInput.GetConfig().last;

                console_cursor_ret_up(2);

                // Overwrite lines
                printf("[%c][%zd/%zd]    %" PRIu64 "(ms)/batch, Speed: %" PRIu64 " KKeys/s   Last key:0x%" PRIX64 " (%u)  Last mutation: %s         \n",
                    WAIT_CHAR(batch), batch, batchesInRound, bruteCallDuration, kResultPerSecond, lastUsedDecryptor.man(), lastUsedDecryptor.seed(), name(configInputMutations[m]));
                console::progressBar(progressPercent, roundTimer.elapsedSeconds());
            }
        }
    }

    console::clearLinesUp(2);

    if (lastResult.hasMatch())
    {
        return getMatchResult(attackRound);
    }

    printf("\n\nAfter: %zd batches no results was found. Keys checked:%zd\n\n", batchesInRound, batchesInRound * keysInBatch);

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
