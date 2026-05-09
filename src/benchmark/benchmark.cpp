#include "benchmark/benchmark.h"

#include <algorithm>
#include <numeric>

#include "common.h"

#include "algorithm/keeloq/keeloq_kernel.h"
#include "algorithm/keeloq/keeloq_encryptor.h"

#include "bruteforce/bruteforce_config.h"
#include "bruteforce/bruteforce_round.h"
#include "bruteforce/bruteforcer.h"
#include "bruteforce/generators/generator_bruteforce.h"

#include "host/console.h"
#include "host/timer.h"


/**
 *  Simple benchmark result struct
 */
struct Result
{
    uint32_t numBlocks;
    uint32_t numThreads;
    int avgSpeed;
};

namespace
{
uint64_t getAvg(const std::vector<uint64_t>& batchesNumKeysPerMs)
{
    if (batchesNumKeysPerMs.empty())
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

std::vector<EncParcel> makeBenchmarkInputs(uint64_t key, uint8_t num, KeeloqLearning::LearningType lType)
{
    // Benchmarks don't need a correct key — we just want something to run the kernels against.
    Encryptor encryptor(key, 987654321);

    std::vector<EncParcel> result;
    result.reserve(num);
    for (uint8_t i = 0; i < num; ++i)
    {
        result.emplace_back(encryptor.click(InputsMutation::None, lType, KeeloqLearning::Modifier::Algo::Normal));
    }
    return result;
}
}

void benchmark::real()
{
    auto dhBruteTime = std::chrono::seconds();
    {
        constexpr uint64_t manDH = 0x8455F43584941223;
        constexpr auto learningDH = KeeloqLearning::LearningType::Simple;

        const std::vector<EncParcel> dhInputs = { 0xFBE31D94BB33DD55, 0xD7577925BB33DD55, 0xAA17DD6CBB33DD55 };

        ScopeTimer timer(&dhBruteTime);
        const auto dhStartDecryptor = Decryptor::MakeNoSeed(manDH & 0xFFFFFFFF00000000);

        Bruteforcer bruteforcer(dhInputs);
        BruteforceConfig simple = BruteforceConfig::GetBruteforce(dhStartDecryptor, InputsMutation::None, 0xFFFFFFFF);

        CudaConfig cudaConfig = CudaConfig::Optimal();

        const auto learningItem = KeeloqLearning::LearningItem(learningDH);
        auto result = bruteforcer.run(simple, cudaConfig, KeeloqLearning::Matrix{ learningItem });

        assert(result.hasMatch() && "Benchmark real test failed, no match found for DH inputs");
        assert(result.match == KeeloqLearning::DecryptedResults::getIndex(learningItem) && "Invalid match for real benchmark test (DH)");

        result.print(dhInputs);
    }
    printf("Real benchmark for DH: Time (s): %" PRIu64 "\n\n", dhBruteTime.count());

    auto sommerBruteTime = std::chrono::seconds();
    {
        constexpr uint64_t manSommer = 0x7BCBEED4376EDCBF;
        constexpr auto learningSommer = KeeloqLearning::LearningType::Normal;

        const std::vector<EncParcel> sommerInputs = { 0x54E9888FBB33DD55, 0x10439539BB33DD55, 0x3210C297BB33DD55 };

        ScopeTimer timer(&sommerBruteTime);
        const auto smStartDecryptor = Decryptor::MakeNoSeed(manSommer & 0xFFFFFFFF00000000);

        Bruteforcer bruteforcer(sommerInputs);
        BruteforceConfig simple = BruteforceConfig::GetBruteforce(smStartDecryptor, InputsMutation::RevKey, 0xFFFFFFFF);

        CudaConfig cudaConfig = CudaConfig::Optimal();

        const auto learningItem = KeeloqLearning::LearningItem(learningSommer);
        auto result = bruteforcer.run(simple, cudaConfig, KeeloqLearning::Matrix({ learningItem }));
        assert(result.hasMatch() && "Benchmark real test failed, no match found for Sommer inputs");
        assert(result.match == KeeloqLearning::DecryptedResults::getIndex(learningItem) && "Invalid match for real benchmark test (Sommer)");

        result.print(sommerInputs);
    }
    printf("Real benchmark for Sommer: Time (s): %" PRIu64 "\n\n", sommerBruteTime.count());
}

int benchmark::run(const std::vector<EncParcel>& inputs, const KeeloqLearning::Matrix& learningMatrix,
    const BruteforceConfig& benchmarkConfig, uint32_t numCudaBlocks, uint16_t numCudaThreads, uint32_t TargetKeysCount)
{
    CudaConfig cudaConfig{ numCudaBlocks, numCudaThreads, 1 };

    BruteforceRound benchmarkRound(inputs, benchmarkConfig, cudaConfig);
    printf("Allocating...");
    benchmarkRound.init();
    printf("\rRunning...   ");

    const auto configInputMutations = benchmarkConfig.getMutations();
    const uint8_t numMutations = static_cast<uint8_t>(configInputMutations.size());

    auto roundTimer = Timer<std::chrono::system_clock>::start();

    const size_t realCalcsInBatch = benchmarkRound.keysPerBatch() * numMutations;
    const size_t numBatches  = std::max<size_t>(3, (TargetKeysCount / realCalcsInBatch) + 1);
    const size_t keysTotal   = numBatches * realCalcsInBatch;

    std::vector<uint64_t> batchesNumKeysPerMs;

    console::setCursorState(false);
    printf("\n");

    bool kernel_fail = false;

    for (size_t i = 0; !kernel_fail && i < numBatches; ++i)
    {
        auto batchTime = std::chrono::microseconds();
        {
            ScopeTimer timer(&batchTime);
            KeeloqKernelInput& kernelInput = benchmarkRound.inputs();
            kernelInput.NextDecryptor();

            auto cudaError = GeneratorBruteforce::PrepareDecryptors(kernelInput, cudaConfig);
            if (cudaError != cudaSuccess)
            {
                printf("Benchmark skipped [%" PRIu32 " x %" PRIu16 " ]. CUDA calculation error: %s: %s\n",
                    numCudaBlocks, numCudaThreads, cudaGetErrorName(cudaError), cudaGetErrorString(cudaError));
                return false;
            }

            for (size_t m = 0; !kernel_fail && m < numMutations; ++m)
            {
                kernelInput.BruteforcePrepare(learningMatrix, configInputMutations[m]);

                auto result = keeloq::kernels::cuda_brute(kernelInput, cudaConfig);
                kernel_fail = benchmarkRound.checkResults(result);

                console_cursor_ret_up(1);
                console::progressBar((i * numMutations + m) / (double)(numBatches * numMutations), roundTimer.elapsedSeconds());

                if (console::readEscPress())
                {
                    console::clearLinesUp(1);
                    printf("Benchmark skipped\n");
                    return 0;
                }
            }
        }

        if (batchTime.count() > 0)
        {
            batchesNumKeysPerMs.push_back(realCalcsInBatch / batchTime.count());
        }
    }

    console::clearLinesUp(1);

    if (kernel_fail)
    {
        printf("| CUDA: %5" PRIu32 " x %-5" PRIu16 " | MKeys:%4" PRIu64 " | GPU Memory %4.1f GB | Round time: FAILURE | Avg.: FAILURE  | Batch avg. speed: FAILURE  | \n",
            numCudaBlocks, numCudaThreads, keysTotal / 1000000,
            benchmarkRound.getMemSize() / static_cast<float>(1024 * 1024 * 1024));

        return -1;
    }

    const uint64_t avg = getAvg(batchesNumKeysPerMs);

    auto roundElapsedMs = roundTimer.elapsed().count();

    auto roundAvgSpeed = keysTotal / roundElapsedMs;

    printf("| CUDA: %5" PRIu32 " x %-5" PRIu16 " | MKeys:%4" PRIu64 " | GPU Memory %4.1f GB | Round time:%6" PRIu64 "ms | Avg.:%4" PRIu64 " Mk/s | Batch avg. speed:%4" PRIu64 " Mk/s | \n",
        numCudaBlocks, numCudaThreads, keysTotal / 1000000,
        benchmarkRound.getMemSize() / static_cast<float>(1024 * 1024 * 1024),
        roundElapsedMs, roundAvgSpeed / 1000,
        avg);

    return static_cast<int>(roundAvgSpeed);
}

void benchmark::run(const std::vector<EncParcel>& inputs, const KeeloqLearning::Matrix& learningMatrix, const BruteforceConfig& benchmarkConfig, uint32_t TargetKeysCount)
{
    static const uint32_t MaxCudaThreads = CudaConfig::MaxCudaThreads();
    static const uint32_t MaxCudaBlocks  = 65536;
    static const float    MaxCudaMemory  = CudaConfig::MaxGlobalMemoryGB();

    const bool hasSeed = benchmarkConfig.hasSeed();

    printf("BENCHMARK BEGIN (Esc to skip)\n\nConfig is: %s\n", BruteforceType::name(benchmarkConfig.type));
    printf(
        "Max Available CUDA Blocks          : %u\n"
        "Max Available GPU Memory           : %.1fGB\n"
        "CUDA Threads per block             : %u\n"
        "Num total calculations             : %u (Millions)\n"
        "Seed specified                     : %s\n"
        "Inputs mutations:                  : %s\n"
        "Learning                           : %s\n\n",
        MaxCudaBlocks, MaxCudaMemory, MaxCudaThreads,
        (TargetKeysCount / 1000000),
        (hasSeed ? "true" : "false"),
        benchmarkConfig.mutationsToString().c_str(),
        learningMatrix.toString(&benchmarkConfig).c_str());

    std::vector<Result> results;

    for (uint32_t numThreads = 256; numThreads <= MaxCudaThreads; numThreads *= 2)
    {
        uint32_t maxBlocksForThreads = std::min(CudaConfig::MaxCudaBlocks(numThreads), MaxCudaBlocks);

        for (uint32_t numBlocks = 1024; numBlocks <= maxBlocksForThreads; numBlocks *= 2)
        {
            auto avgSpeed = run(inputs, learningMatrix, benchmarkConfig, numBlocks, numThreads, TargetKeysCount);
            if (!avgSpeed)
            {
                // Cancel request
                return;
            }

            results.push_back({ numBlocks, numThreads, static_cast<int>(avgSpeed) });
        }
    }

    std::sort(results.begin(), results.end(), [](const Result& a, const Result& b) { return a.avgSpeed > b.avgSpeed; });

    printf("\nBENCHMARK FINISHED: Best result: %u blocks, %u threads, %.3f million keys/s\n\n\n",
        results.front().numBlocks, results.front().numThreads, results.front().avgSpeed / 1000.0f);
}

void benchmark::all(const CommandLineArgs& /*args*/)
{
#if _DEBUG
    constexpr size_t TargetCalculationsNumber = 10000000;
#else
    constexpr size_t TargetCalculationsNumber = 100000000;
#endif

    using namespace KeeloqLearning;
    console_clear();

    constexpr auto NumInputs = 3;
    auto inputs = makeBenchmarkInputs(0xFF123FF3434FFFFF, NumInputs, LearningType::Serial3);


    {
        real();
    }

    {
        auto benchConfigSimpleBrute = BruteforceConfig::GetBruteforce(
            Decryptor::Make(0, 1234567, true), InputsMutation::None, 1);

        // EASIEST: Single easiest brute type only
        auto learningMatrix = KeeloqLearning::Matrix{ LearningType::Simple };
        run(inputs, learningMatrix, benchConfigSimpleBrute, TargetCalculationsNumber);

        // HARDEST: Full brute 11 x 2 x 4 permutations
        learningMatrix = KeeloqLearning::Matrix::Everything();
        benchConfigSimpleBrute.setMutationMask(InputsMutation::All);
        run(inputs, learningMatrix, benchConfigSimpleBrute, TargetCalculationsNumber / 10);
    }

    {
        auto configXorFixed = BruteforceConfig::GetXorFixBruteforce(Decryptor::Make(0, 0, true), InputsMutation::RevKey, 1);

        // XORed only
        auto learningMatrix = KeeloqLearning::Matrix::Everything();
        run(inputs, learningMatrix, configXorFixed, TargetCalculationsNumber);
    }

    {
        auto configSeedOnly = BruteforceConfig::GetSeedBruteforce(Decryptor::Make(0, 0, true), InputsMutation::None, 1);

        // Seed only
        auto learningMatrix = KeeloqLearning::Matrix{ LearningType::Secure };
        run(inputs, learningMatrix, configSeedOnly, TargetCalculationsNumber);
    }


    {
        // Alphabet brute benchmarks
        auto configNoSeed = BruteforceConfig::GetAlphabet(Decryptor::MakeNoSeed(0), InputsMutation::None, "0123456789abcdefgh"_b);
        auto configSeeded = BruteforceConfig::GetAlphabet(Decryptor::Make(0, 1234567, true), InputsMutation::None, "0123456789abcdefgh"_b);

        auto learningMatrix = KeeloqLearning::Matrix{ LearningType::Simple };
        run(inputs, learningMatrix, configSeeded, TargetCalculationsNumber);
        run(inputs, learningMatrix, configNoSeed, TargetCalculationsNumber);

        learningMatrix = KeeloqLearning::Matrix{ LearningType::Secure };
        run(inputs, learningMatrix, configSeeded, TargetCalculationsNumber);

        learningMatrix = KeeloqLearning::Matrix{ LearningType::Faac };
        run(inputs, learningMatrix, configSeeded, TargetCalculationsNumber);

        learningMatrix = KeeloqLearning::Matrix{ LearningType::Simple, LearningType::Normal, LearningType::Xor };
        run(inputs, learningMatrix, configNoSeed, TargetCalculationsNumber);
    }
}
