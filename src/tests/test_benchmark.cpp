#include "common.h"

#include <numeric>
#include <algorithm>

#include "test_benchmark.h"
#include "host/console.h"
#include "host/timer.h"

#include "bruteforce/bruteforcer.h"
#include "bruteforce/generators/generator_bruteforce.h"
#include "bruteforce/bruteforce_config.h"
#include "bruteforce/bruteforce_round.h"

#include "algorithm/keeloq/keeloq_kernel.h"
#include "tests/test_keeloq.h"

#if _DEBUG
constexpr size_t TargetCalculations = 10000000;
#else
constexpr size_t TargetCalculations = 100000000;
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

bool benchmark::run(const std::vector<EncParcel>& inputs, const KeeloqLearning::Matrix& learningMatrix, const BruteforceConfig& benchmarkConfig,
    uint16_t numCudaBlocks, uint16_t numCudaThreads)
{
    CudaConfig cudaConfig{ numCudaBlocks, numCudaThreads, 1 };

    BruteforceRound benchmarkRound(inputs, benchmarkConfig, learningMatrix, cudaConfig);
    benchmarkRound.Init();

    // Allocations time above doesn't count
    auto roundTimer = Timer<std::chrono::system_clock>::start();

    const size_t keysInBatch = benchmarkRound.keys_per_batch();
    const size_t numBatches = std::max<size_t>(4, (TargetCalculations / keysInBatch) + 1);
    const size_t keysTotal = numBatches * keysInBatch;

    std::vector<uint64_t> batchesNumKeysPerMs;

    console::set_cursor_state(false);
    printf("\n");

    bool kernel_fail = false;

    for (size_t i = 0; !kernel_fail && i < numBatches; ++i)
    {
        auto batchTime = std::chrono::milliseconds();
        {
            ScopeTimer timer(&batchTime);
            KeeloqKernelInput& kernelInput = benchmarkRound.Inputs();
            kernelInput.NextDecryptor();

            auto cudaError = GeneratorBruteforce::PrepareDecryptors(kernelInput, cudaConfig);
            if (cudaError != cudaSuccess)
            {
                printf("Benchmark skipped [%" PRIu16 " x %" PRIu16 " ]. CUDA calculation error: %s: %s\n",
                    numCudaBlocks, numCudaThreads, cudaGetErrorName(cudaError), cudaGetErrorString(cudaError));
                return false;
            }

            auto result = keeloq::kernels::cuda_brute(kernelInput, cudaConfig);
            kernel_fail = benchmarkRound.check_results(result);
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

    if (!kernel_fail)
    {
        const uint64_t avg = getAvg(batchesNumKeysPerMs);

        console_cursor_ret_up(1);
        console::clear_line();

        auto roundElapsedMs = roundTimer.elapsed().count();
        auto roundAvgSpeed = keysTotal / roundElapsedMs;

        // Creating results
        printf("| CUDA: %5" PRIu16 " x %-5" PRIu16 " | MKeys: %3" PRIu64 " | GPU Memory (MB):%-6" PRIu64 " | Round: Time (ms):%-6" PRIu64 " | Avg. speed (K/s):%-8" PRIu64 " | Batch Avg. Speed (K/s):%-8" PRIu64 " | \n",
            numCudaBlocks, numCudaThreads, keysTotal / 1000000,
            benchmarkRound.get_mem_size() / (1024 * 1024),
            roundElapsedMs, roundAvgSpeed,
            avg);
    }
    else
    {
        printf("| CUDA: %5" PRIu16 " x %-5" PRIu16 " | MKeys: %3" PRIu64 " | GPU Memory (MB):%-6" PRIu64 " | Round: Time (ms):FAILURE| Avg. speed (K/s):FAILURE  | Batch Avg. Speed (K/s):FAILURE   | \n",
            numCudaBlocks, numCudaThreads, keysTotal / 1000000,
            benchmarkRound.get_mem_size() / (1024 * 1024));
    }

    return true;
}


void benchmark::real()
{
    constexpr uint64_t manDH = 0x8455F43584941223;
    constexpr auto learningDH = KeeloqLearning::LearningType::Simple;

    constexpr uint64_t manSommer = 0x7BCBEED4376EDCBF;
    constexpr auto learningSommer = KeeloqLearning::LearningType::Normal;

    const std::vector<EncParcel> dhInputs       = { 0xFBE31D94BB33DD55, 0xD7577925BB33DD55, 0xAA17DD6CBB33DD55 }; //CNT: 2254
    const std::vector<EncParcel> sommerInputs   = { 0x54E9888FBB33DD55, 0x10439539BB33DD55, 0x3210C297BB33DD55 }; //CNT: 0019

    auto dhBruteTime = std::chrono::seconds();
    {
        ScopeTimer timer(&dhBruteTime);

        Bruteforcer bruteforcer(dhInputs);
        BruteforceConfig simple = BruteforceConfig::GetBruteforce(Decryptor::MakeNoSeed(manDH & 0xFFFFFFFF00000000), 0xFFFFFFFF);

        CudaConfig cudaConfig = CudaConfig::Optimal();

        const auto learningItem = KeeloqLearning::LearningItem(learningDH);

        auto result = bruteforcer.run(simple, cudaConfig, KeeloqLearning::Matrix({ learningItem }));
        assert(result.hasMatch() && "Benchmark real test failed, no match found for DH inputs");

        assertf(result.match == KeeloqLearning::DecryptedResults::getIndex(learningItem),
            "Invalid match for real benchmark test, expected: %s (%s, %s), got: %s (%s, %s)",
            KeeloqLearning::Name(learningDH), KeeloqLearning::Name(learningItem.imod), KeeloqLearning::Name(learningItem.amod),
            KeeloqLearning::Name(KeeloqLearning::DecryptedResults::getByIndex(result.match).learning),
            KeeloqLearning::Name(KeeloqLearning::DecryptedResults::getByIndex(result.match).imod),
            KeeloqLearning::Name(KeeloqLearning::DecryptedResults::getByIndex(result.match).amod));

        result.print(dhInputs);
    }
    printf("Real benchmark for DH: Time (s): %" PRIu64 "\n\n", dhBruteTime.count());

    auto sommerBruteTime = std::chrono::seconds();
    {
        ScopeTimer timer(&sommerBruteTime);

        Bruteforcer bruteforcer(sommerInputs);
        BruteforceConfig simple = BruteforceConfig::GetBruteforce(Decryptor::MakeNoSeed(manSommer & 0xFFFFFFFF00000000), 0xFFFFFFFF);

        CudaConfig cudaConfig = CudaConfig::Optimal();

        const auto learningItem = KeeloqLearning::LearningItem(learningSommer);
        auto result = bruteforcer.run(simple, cudaConfig, KeeloqLearning::Matrix({ learningItem }));
        assert(result.hasMatch() && "Benchmark real test failed, no match found for DH inputs");

        assertf(result.match == KeeloqLearning::DecryptedResults::getIndex(learningItem),
            "Invalid match for real benchmark test, expected: %s (%s, %s), got: %s (%s, %s)",
            KeeloqLearning::Name(learningSommer), KeeloqLearning::Name(learningItem.imod), KeeloqLearning::Name(learningItem.amod),
            KeeloqLearning::Name(KeeloqLearning::DecryptedResults::getByIndex(result.match).learning),
            KeeloqLearning::Name(KeeloqLearning::DecryptedResults::getByIndex(result.match).imod),
            KeeloqLearning::Name(KeeloqLearning::DecryptedResults::getByIndex(result.match).amod));

        result.print(sommerInputs);
    }
    printf("Real benchmark for Sommer: Time (s): %" PRIu64 "\n\n", sommerBruteTime.count());
}

void benchmark::run(const std::vector<EncParcel>& inputs, const KeeloqLearning::Matrix& learningMatrix, const BruteforceConfig& benchmarkConfig)
{
    static const uint32_t MaxCudaThreads = CudaConfig::MaxCudaThreads();
    static const uint32_t MaxCudaBlocks = CudaConfig::MaxCudaBlocks();
    static const float MaxCudaMemory = CudaConfig::MaxGlobalMemoryGB();

    const bool hasSeed = benchmarkConfig.has_seed();

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
        (hasSeed ? "true" : "false"),
        learningMatrix.to_string(&benchmarkConfig).c_str());


    for (uint32_t numBlocks = 1024; numBlocks <= MaxCudaBlocks; numBlocks *= 2)
    {
        for (uint32_t numThreads = 256; numThreads <= MaxCudaThreads; numThreads *= 2)
        {
            if (!run(inputs, learningMatrix, benchmarkConfig, numBlocks, numThreads))
            {
                return;
            }
        }
    }

    printf("\nBENCHMARK FINISHED\n\n\n");
}

void benchmark::all(const CommandLineArgs& args)
{
    using namespace KeeloqLearning;
    console_clear();

    // Real test first
    real();

    BruteforceConfig benchmarkConfig_no_seed = BruteforceConfig::GetAlphabet(Decryptor::MakeNoSeed(0), "0123456789abcdefgh"_b);
    BruteforceConfig benchmarkConfig_wt_seed = BruteforceConfig::GetAlphabet(Decryptor::Make(0, 1234567, true), "0123456789abcdefgh"_b);
    BruteforceConfig benchmarkConfig_simple = BruteforceConfig::GetBruteforce(Decryptor::Make(0, 1234567, true), 1);
    BruteforceConfig benchmarkConfig_only_seed = BruteforceConfig::GetSeedBruteforce(Decryptor::Make(0, 0, true), 1);


    // Any inputs, we don't need to find the key
    constexpr auto NumInputs = 3;
    auto inputs = tests::keeloq::gen_inputs(0xFF123FF3434FFFFF, NumInputs, LearningType::Serial3);

    auto learningMatrix = KeeloqLearning::Matrix({ LearningItem(LearningType::Simple, Modifier::Input::Normal, Modifier::Algo::Normal) });
    run(inputs, learningMatrix, benchmarkConfig_simple);

    learningMatrix = KeeloqLearning::Matrix({ LearningType::Simple });
    run(inputs, learningMatrix, benchmarkConfig_wt_seed);
    run(inputs, learningMatrix, benchmarkConfig_no_seed);

    learningMatrix = KeeloqLearning::Matrix({ LearningType::Secure });
    run(inputs, learningMatrix, benchmarkConfig_only_seed);

    learningMatrix = KeeloqLearning::Matrix({ LearningType::Simple });
    run(inputs, learningMatrix, benchmarkConfig_no_seed);
    run(inputs, learningMatrix, benchmarkConfig_simple);

    learningMatrix = KeeloqLearning::Matrix({ LearningType::Normal });
    run(inputs, learningMatrix, benchmarkConfig_no_seed);

    learningMatrix = KeeloqLearning::Matrix({ LearningType::Secure });
    run(inputs, learningMatrix, benchmarkConfig_wt_seed);

    learningMatrix = KeeloqLearning::Matrix({ LearningType::Faac });
    run(inputs, learningMatrix, benchmarkConfig_wt_seed);

    learningMatrix = KeeloqLearning::Matrix({ LearningType::Simple, LearningType::Normal, LearningType::Xor });
    run(inputs, learningMatrix, benchmarkConfig_no_seed);
}
