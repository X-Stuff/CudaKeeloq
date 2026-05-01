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

#if _DEBUG
constexpr size_t TargetCalculations = 10000000;
#else
constexpr size_t TargetCalculations = 500000000;
#endif

namespace
{
uint64_t getAvg(const std::vector<uint64_t>& batchesNumKeysPerMs)
{
    if (batchesNumKeysPerMs.size() == 0)
    {
        return 0;
    }

    uint64_t avg = 0;
    for (const auto& num : batchesNumKeysPerMs)
    {
        avg += num;
    }

    return avg / batchesNumKeysPerMs.size();
}
}

bool benchmark::run(const CommandLineArgs& args, const BruteforceConfig& benchmarkConfig, uint16_t numCudaBlocks, uint16_t numCudaThreads)
{
    auto learningMatrix = KeeloqLearning::Matrix(args.selected_learning, args.selected_mod_mask);

    // Allocations doesn't count
    BruteforceRound benchmarkRound(args.inputs, benchmarkConfig, learningMatrix, numCudaBlocks, numCudaThreads, args.cuda_loops);
    benchmarkRound.Init();

    auto roundTimer = Timer<std::chrono::system_clock>::start();

    const size_t keysInBatch = benchmarkRound.keys_per_batch();
    const size_t numBatches = std::max<size_t>(4, (TargetCalculations / keysInBatch) + 1);
    const size_t keysTotal = numBatches * keysInBatch;

    std::vector<uint64_t> batchesNumKeysPerMs;

    console::set_cursor_state(false);
    printf("\n");

    for (size_t i = 0; i < numBatches; ++i)
    {
        auto batchTime = std::chrono::milliseconds();
        {
            ScopeTimer timer(&batchTime);
            KeeloqKernelInput& kernelInput = benchmarkRound.Inputs();
            kernelInput.NextDecryptor();

            GeneratorBruteforce::PrepareDecryptors(kernelInput, numCudaBlocks, numCudaThreads);
            auto result = keeloq::kernels::cuda_brute(kernelInput, numCudaBlocks, numCudaThreads);
            if (result.error)
            {
                printf("Benchmark skipped. CUDA calculation error: %" PRIu16 "x %" PRIu16 "\n",
                    numCudaBlocks, numCudaThreads);
                return false;
            }
        }

        if (batchTime.count() > 0)
        {
            batchesNumKeysPerMs.push_back(keysInBatch / batchTime.count());
        }

        console_cursor_ret_up(1);
        console::progress_bar(i / (double)numBatches, roundTimer.elapsed_seconds());

        if (console::read_esc_press())
        {
            console_cursor_ret_up(1);
            console::clear_line();
            printf("Benchmark skipped\n");
            return false;
        }
    }

    const uint64_t avg = getAvg(batchesNumKeysPerMs);

    console_cursor_ret_up(1);
    console::clear_line();

    auto roundElapsedMs = roundTimer.elapsed().count();
    auto roundAvgSpeed = keysTotal / roundElapsedMs;

    // Creating results
    printf("| CUDA: %5" PRIu16 " x %-5" PRIu16 " | MKeys: %3" PRIu64 " | GPU Memory (MB):%-6" PRIu64 " | Round: Time (ms):%-6" PRIu64 " | Avg. speed (K/s):%-7" PRIu64 " | Batch Avg. Speed (K/s):%-7" PRIu64 " | \n",
        numCudaBlocks, numCudaThreads, keysTotal / 1000000,
        benchmarkRound.get_mem_size() / (1024 * 1024),
        roundElapsedMs, roundAvgSpeed,
        avg);

    return true;
}

void benchmark::run(const CommandLineArgs& args, const BruteforceConfig& benchmarkConfig)
{
    static const uint32_t MaxCudaThreads = CommandLineArgs::max_cuda_threads();
    static const uint32_t MaxCudaBlocks = CommandLineArgs::max_cuda_blocks();
    static const float MaxCudaMemory = (CommandLineArgs::max_global_memory() / static_cast<float>(1024 * 1024 * 1024));

    printf("BENCHMARK BEGIN (Esc to skip)\n\nConfig is: %s\n", BruteforceType::Name(benchmarkConfig.type));
#ifndef NO_INNER_LOOPS
    printf("Num loops inside CUDA           : %u\n", args.cuda_loops);
#endif // !NO_INNER_LOOPS
    printf(
        "Max Available CUDA Blocks          : %u\n"
        "Max Available GPU Memory           : %.1fGB\n"
        "CUDA Threads per block             : %u\n"
        "Num total calculations             : %" PRIu64 " (Millions)\n"
        "Seed specified                     : %s\n"
        "Learning                           : %s\n\n",
        MaxCudaBlocks, MaxCudaMemory, MaxCudaThreads,
        (TargetCalculations / 1000000),
        (benchmarkConfig.start.seed() == 0 ? "false" : "true"),
        KeeloqLearning::Matrix(args.selected_learning, args.selected_mod_mask).to_string().c_str());


    for (uint32_t numBlocks = 256; numBlocks <= MaxCudaBlocks; numBlocks *= 2)
    {
        if (!run(args, benchmarkConfig, numBlocks, MaxCudaThreads))
        {
            return;
        }
    }

    printf("\nBENCHMARK FINISHED\n\n\n");
}

void benchmark::all(const CommandLineArgs& args)
{
    using namespace KeeloqLearning;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    BruteforceConfig benchmarkConfig_no_seed = BruteforceConfig::GetAlphabet(Decryptor::MakeNoSeed(0), "0123456789abcdefgh"_b);
    BruteforceConfig benchmarkConfig_wt_seed = BruteforceConfig::GetAlphabet(Decryptor::Make(0, 1234567, true), "0123456789abcdefgh"_b);
    BruteforceConfig benchmarkConfig_simple = BruteforceConfig::GetBruteforce(Decryptor::Make(0, 1234567, true), 1);

    console_clear();
    CommandLineArgs copy = args;

    // Any inputs, we don't need to find the key
    constexpr auto NumInputs = 3;
    copy.inputs = tests::keeloq::gen_inputs(0xFF123FF3434FFFFF, NumInputs, LearningType::Serial3);

    copy.selected_learning = {};
    run(copy, benchmarkConfig_simple);
    run(copy, benchmarkConfig_wt_seed);
    run(copy, benchmarkConfig_no_seed);

    copy.selected_learning = { LearningType::Simple };
    run(copy, benchmarkConfig_no_seed);

    copy.selected_learning = { LearningType::Normal };
    run(copy, benchmarkConfig_no_seed);

    copy.selected_learning = { LearningType::Secure };
    run(copy, benchmarkConfig_wt_seed);

    copy.selected_learning = { LearningType::Faac };
    run(copy, benchmarkConfig_wt_seed);

    copy.selected_learning = { LearningType::Simple, LearningType::Normal, LearningType::Xor };
    run(copy, benchmarkConfig_no_seed);

#ifndef NO_INNER_LOOPS
    copy.cuda_loops = 4;
    run(copy);
#endif
}
