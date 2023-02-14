#include "common.h"

#include <numeric>

#include "test_benchmark.h"
#include "host/console.h"

#include "bruteforce/generators/generator_bruteforce.h"
#include "bruteforce/bruteforce_config.h"
#include "bruteforce/bruteforce_round.h"

#include "algorithm/keeloq/keeloq_kernel.h"


void benchmark::run(const CommandLineArgs& args)
{
#if _DEBUG
    constexpr size_t TargetCalculations = 10000000;
#else
    constexpr size_t TargetCalculations = 100000000;
#endif

    constexpr uint16_t CudaBlocks[] = { 128, 256, 512, 1024, 2048, 4096, 8196 };
    constexpr uint16_t CudaThreads[] = { 128, 256, 512, 1024, 2048 };

    static const uint32_t MaxCudaThreads = CommandLineArgs::max_cuda_threads();
    static const uint32_t MaxCudaBlocks = CommandLineArgs::max_cuda_blocks();

    BruteforceConfig benchmarkConfig = BruteforceConfig::GetAlphabet(0, "0123456789abcdefgh"_b);

    printf("BENCHMARK BEGIN\n\nConfig is: Alphabet\n");
#ifndef NO_INNER_LOOPS
    printf("Num loops inside CUDA\t\t\t: %u (from cmd)\n", args.cuda_loops);
#endif // !NO_INNER_LOOPS
    printf("Max Available CUDA Threads per block\t: %u\n"
        "Max Available CUDA Blocks\t\t: %u\n"
        "Num total calculations\t\t\t: %" PRIu64 " (Millions)\n"
        "Learning (from cmd)\t\t\t: %s\n\n",
        MaxCudaThreads, MaxCudaBlocks,
        (TargetCalculations / 1000000),
        KeeloqLearningType::to_string(args.selected_learning).c_str());


    bool in_progress = true;
    for (auto NumCudaBlocks : CudaBlocks)
    {
        if (!in_progress || NumCudaBlocks > MaxCudaBlocks)
        {
            break;
        }

        for (auto NumCudaThreads : CudaThreads)
        {
            if (!in_progress || NumCudaThreads > MaxCudaThreads)
            {
                break;
            }

            BruteforceRound benchmarkRound(args.inputs, benchmarkConfig, args.selected_learning, NumCudaBlocks, NumCudaThreads, args.cuda_loops);
            benchmarkRound.Init();

            size_t keysInBatch = benchmarkRound.keys_per_batch();
            size_t numBatches = std::max<size_t>(1, TargetCalculations / keysInBatch);

            auto roundStartTime = std::chrono::system_clock::now();

            std::vector<uint64_t> batches_kResults_per_sec(numBatches);

            console::set_cursor_state(false);
            printf("\n");

            for (size_t i = 0; in_progress && i < numBatches; ++i)
            {
                auto batchStartTime = std::chrono::high_resolution_clock::now();

                KeeloqKernelInput& kernelInput = benchmarkRound.Inputs();
                kernelInput.NextDecryptor();

                GeneratorBruteforce::PrepareDecryptors(kernelInput, NumCudaBlocks, NumCudaThreads);
                keeloq::kernels::cuda_brute(kernelInput, NumCudaBlocks, NumCudaThreads);

                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::high_resolution_clock::now() - batchStartTime);

                batches_kResults_per_sec[i] = duration.count() == 0 ? 0 : keysInBatch / duration.count();

                auto overall = std::chrono::duration_cast<std::chrono::seconds>(
                    std::chrono::system_clock::now() - roundStartTime);

                console_cursor_ret_up(1);
                console::progress_bar(i / (double)numBatches, overall);

                if (console::read_esc_press())
                {
                    console_cursor_ret_up(1);
                    console::clear_line();
                    printf("Benchmark skipped\n");
                    in_progress = false;
                }
            }

            if (in_progress)
            {
                auto overall = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::system_clock::now() - roundStartTime);

                std::sort(batches_kResults_per_sec.begin(), batches_kResults_per_sec.end());

                console_cursor_ret_up(1);
                console::clear_line();

                // Creating results
                printf("| CUDA: %" PRIu16 " x %" PRIu16 " \t| MEM: %" PRIu64 " MB\t | Time (ms): %" PRIu64 " \t |\tSpeed (K/s): %" PRIu64 " (avg.) %" PRIu64 " (mean) |\t\t\t\t\n",
                    NumCudaBlocks, NumCudaThreads,
                    benchmarkRound.get_mem_size() / (1024 * 1024),
                    overall.count(),
                    std::reduce(batches_kResults_per_sec.begin(), batches_kResults_per_sec.end()) / numBatches,
                    batches_kResults_per_sec[numBatches / 2]);
            }
        }
    }
}

void benchmark::all(const CommandLineArgs& args)
{
    console_clear();
    CommandLineArgs copy = args;
    copy.inputs = { 0, 1, 2 };

    copy.cuda_loops = 2;
    copy.selected_learning = {};
    run(copy);

    copy.cuda_loops = 2;
    copy.selected_learning = { KeeloqLearningType::Simple };
    run(copy);

#ifndef NO_INNER_LOOPS
    copy.cuda_loops = 4;
    run(copy);
#endif
}
