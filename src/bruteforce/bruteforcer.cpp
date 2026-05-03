#include "bruteforce/bruteforcer.h"
#include "bruteforce/bruteforce_round.h"

#include "algorithm/keeloq/keeloq_kernel.h"
#include "algorithm/keeloq/keeloq_learning_types.h"

#include "generators/generator_bruteforce.h"

#include "host/console.h"
#include "host/command_line_args.h"
#include "host/timer.h"

#include <chrono>


Bruteforcer::Bruteforcer(const std::vector<EncParcel>& inputs) : inputs(inputs)
{

}

SingleResult Bruteforcer::run(const BruteforceConfig& config, const CudaConfig& cuda, const KeeloqLearning::Matrix& learningMatrix)
{
    BruteforceRound attackRound(inputs, config, learningMatrix, cuda);

    printf("\nAllocating...");
    attackRound.Init();

    printf("\rRunning...    \n%s\n", attackRound.to_string().c_str());

    KernelResult lastResult;

    const size_t batchesInRound = attackRound.num_batches();
    const size_t keysInBatch = attackRound.keys_per_batch();

    auto roundTimer = Timer<std::chrono::steady_clock>::start();

    for (size_t batch = 0; batch < batchesInRound; ++batch)
    {
        auto batchTimer = Timer<std::chrono::steady_clock>::start();

        KeeloqKernelInput& kernelInput = attackRound.Inputs();

        if (attackRound.Type() != BruteforceType::Dictionary)
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

        // do the bruteforce
        lastResult = keeloq::kernels::cuda_brute(kernelInput, cuda);
        if (attackRound.check_results(lastResult))
        {
            break;
        }

        if (batch == 0)
        {
            console::set_cursor_state(false);
            printf("\n\n\n");
        }

        const auto batchDurationMs = batchTimer.elapsed().count();

        const auto kResultPreSecond = batchDurationMs == 0 ? 0 : keysInBatch / batchDurationMs;
        const auto progressPercent = (double)(batch + 1) / batchesInRound;

        const Decryptor& lastUsedDecryptor = kernelInput.GetConfig().last;

        console_cursor_ret_up(2);

        // Overwrite lines
        printf("[%c][%zd/%zd]    %" PRIu64 "(ms)/batch Speed: %" PRIu64 " KKeys/s   Last key:0x%" PRIX64 " (%u)           \n",
            WAIT_CHAR(batch), batch, batchesInRound, batchDurationMs, kResultPreSecond, lastUsedDecryptor.man(), lastUsedDecryptor.seed());
        console::progress_bar(progressPercent, roundTimer.elapsed_seconds());
    }

    console::clear_lines_up(2);

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

    const auto results = round.Inputs().results->host();

    auto searchTimer = Timer<std::chrono::steady_clock>::start();

    const auto numIterations = results.num / MaxElements;

    for (size_t index = 0, count = 0; index < results.num; index += MaxElements, ++count)
    {
        // Progress bar
        console_cursor_ret_up(2);

        printf("[%c][%zd/%zd] Searching match in GPU memory...                                                                 \n",
            WAIT_CHAR(count), count, numIterations);

        const auto progressPercent = (double)(count + 1) / numIterations;
        console::progress_bar(progressPercent, searchTimer.elapsed_seconds());


        // Read decryptors in batches, just to save RAM on host, using static method since we already copied object to host
        auto copied_results = CudaArray<SingleResult>::read(results, index, MaxElements);

        for (const auto& result : copied_results)
        {
            if (result.match != KeeloqLearning::NoMatch)
            {
                console::clear_lines_up(2);
                printf("Found!\n");
                return result;
            }
        }
    }

    console::clear_lines_up(2);
    printf("NOT FOUND!\n");
    return SingleResult::Invalid();
}
