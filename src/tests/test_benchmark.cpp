#include "common.h"

#include <numeric>
#include <algorithm>

#include "test_benchmark.h"
#include "host/console.h"
#include "host/timer.h"

#include "bruteforce/generators/generator_bruteforce.h"
#include "bruteforce/bruteforce_config.h"
#include "bruteforce/bruteforce_round.h"

#include "algorithm/keeloq/keeloq_kernel.h"
#include "tests/test_keeloq.h"


void benchmark::run(const CommandLineArgs& args, const BruteforceConfig& benchmarkConfig, const std::vector<uint16_t>& CudaBlocks, const std::vector<uint16_t>& CudaThreads)
{
#if _DEBUG
    constexpr size_t TargetCalculations = 10000000;
#else
    constexpr size_t TargetCalculations = 100000000;
#endif

    static const uint32_t MaxCudaThreads = CommandLineArgs::max_cuda_threads();
    static const uint32_t MaxCudaBlocks = CommandLineArgs::max_cuda_blocks();

    printf("BENCHMARK BEGIN\n\nConfig is: Alphabet\n");
#ifndef NO_INNER_LOOPS
    printf("Num loops inside CUDA                   : %u\n", args.cuda_loops);
#endif // !NO_INNER_LOOPS
    printf(
        "Max Available CUDA Threads per block       : %u\n"
        "Max Available CUDA Blocks                  : %u\n"
        "Num total calculations                     : %" PRIu64 " (Millions)\n"
        "Learning                                   : %s\n"
        "Seed specified                             : %s\n\n",
        MaxCudaThreads, MaxCudaBlocks,
        (TargetCalculations / 1000000),
        KeeloqLearningType::to_string(args.selected_learning).c_str(),
        (benchmarkConfig.start.seed() == 0 ? "false" : "true"));


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

            std::vector<uint64_t> batches_kResults_per_sec;

            console::set_cursor_state(false);
            printf("\n");

            auto roundTimer = Timer<std::chrono::system_clock>::start();

            for (size_t i = 0; in_progress && i < numBatches; ++i)
            {
                auto batchTimer = Timer<std::chrono::high_resolution_clock>::start();

                KeeloqKernelInput& kernelInput = benchmarkRound.Inputs();
                kernelInput.NextDecryptor();

                GeneratorBruteforce::PrepareDecryptors(kernelInput, NumCudaBlocks, NumCudaThreads);
                keeloq::kernels::cuda_brute(kernelInput, NumCudaBlocks, NumCudaThreads);

                auto elapsedMs = batchTimer.elapsed().count();
                if (elapsedMs > 0)
                {
                    batches_kResults_per_sec.push_back(keysInBatch / elapsedMs);
                }

                console_cursor_ret_up(1);
                console::progress_bar(i / (double)numBatches, roundTimer.elapsed_secods());

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
                std::sort(batches_kResults_per_sec.begin(), batches_kResults_per_sec.end());

                console_cursor_ret_up(1);
                console::clear_line();

                // Creating results
                printf("| CUDA: %" PRIu16 " x %" PRIu16 " \t| MEM: %" PRIu64 " MB\t | Time (ms): %" PRIu64 " \t |\tSpeed (K/s): %" PRIu64 " (avg.) %" PRIu64 " (mean) |\t\t\t\t\n",
                    NumCudaBlocks, NumCudaThreads,
                    benchmarkRound.get_mem_size() / (1024 * 1024),
                    roundTimer.elapsed().count(),
                    std::reduce(batches_kResults_per_sec.begin(), batches_kResults_per_sec.end()) / numBatches,
                    batches_kResults_per_sec[numBatches / 2]);
            }
        }
    }
}

void benchmark::all(const CommandLineArgs& args)
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    std::vector<uint16_t> CudaBlocks  = { 256, 512, 1024, 2048, 4096, 8196 };
    std::vector<uint16_t> CudaThreads = { 128, 256, 512, 1024, 2048 };

    BruteforceConfig benchmarkConfig_no_seed = BruteforceConfig::GetAlphabet(Decryptor(0, 0), "0123456789abcdefgh"_b);
    BruteforceConfig benchmarkConfig_wt_seed = BruteforceConfig::GetAlphabet(Decryptor(0, 1234567), "0123456789abcdefgh"_b);

    console_clear();
    CommandLineArgs copy = args;
    copy.inputs = tests::keeloq::gen_inputs(0xFF123FF3434FFFFF);

    copy.selected_learning = {};
    run(copy, benchmarkConfig_wt_seed, CudaBlocks, CudaThreads);
    run(copy, benchmarkConfig_no_seed, CudaBlocks, CudaThreads);

    copy.selected_learning = { KeeloqLearningType::Simple };
    run(copy, benchmarkConfig_no_seed, CudaBlocks, CudaThreads);

    copy.selected_learning = { KeeloqLearningType::Normal };
    run(copy, benchmarkConfig_no_seed, CudaBlocks, CudaThreads);

    copy.selected_learning = { KeeloqLearningType::Secure };
    run(copy, benchmarkConfig_wt_seed, CudaBlocks, CudaThreads);

    copy.selected_learning = { KeeloqLearningType::Faac };
    run(copy, benchmarkConfig_wt_seed, CudaBlocks, CudaThreads);

    copy.selected_learning = { KeeloqLearningType::Simple, KeeloqLearningType::Normal, KeeloqLearningType::Xor};
    run(copy, benchmarkConfig_no_seed, CudaBlocks, CudaThreads);

#ifndef NO_INNER_LOOPS
    copy.cuda_loops = 4;
    run(copy, CudaBlocks, CudaThreads);
#endif
}
