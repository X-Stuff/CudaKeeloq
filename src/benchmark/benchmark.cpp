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
    {
        constexpr uint64_t manDH = 0x8455F43584941223;
        constexpr auto learningDH = KeeloqLearning::LearningType::Simple;

        const std::vector<EncParcel> dhInputs = { 0xFBE31D94BB33DD55, 0xD7577925BB33DD55, 0xAA17DD6CBB33DD55 };

        auto timer = Timer<std::chrono::steady_clock>::start();
        const auto dhStartDecryptor = Decryptor::MakeNoSeed(manDH & 0xFFFFFFFF00000000);

        Bruteforcer bruteforcer(dhInputs, true);
        BruteforceConfig simple = BruteforceConfig::GetBruteforce(dhStartDecryptor, InputsMutation::None, 0xFFFFFFFF);

        CudaConfig cudaConfig = CudaConfig::Optimal();

        const auto learningItem = KeeloqLearning::LearningItem(learningDH);
        auto result = bruteforcer.run(simple, cudaConfig, KeeloqLearning::Matrix{ learningItem });
        const auto& stats = bruteforcer.getStats();

        if (stats.result != Bruteforcer::Stats::UserCancelled)
        {
            assert(stats.result == Bruteforcer::Stats::Success && "Benchmark real test failed, no successful result");
            assert(result.hasMatch() && "Benchmark real test failed, no match found for DH inputs");
            assert(result.match == KeeloqLearning::DecryptedResults::getIndex(learningItem) && "Invalid match for real benchmark test (DH)");

            result.print(dhInputs);
            printf("Real benchmark for DH: Time (s): %" PRIu64 "\n\n", timer.elapsedSeconds().count());
        }
    }

    {
        constexpr uint64_t manSommer = 0x7BCBEED4376EDCBF;
        constexpr auto learningSommer = KeeloqLearning::LearningType::Normal;

        const std::vector<EncParcel> sommerInputs = { 0x54E9888FBB33DD55, 0x10439539BB33DD55, 0x3210C297BB33DD55 };

        auto timer = Timer<std::chrono::steady_clock>::start();
        const auto smStartDecryptor = Decryptor::MakeNoSeed(manSommer & 0xFFFFFFFF00000000);

        Bruteforcer bruteforcer(sommerInputs, true);
        BruteforceConfig simple = BruteforceConfig::GetBruteforce(smStartDecryptor, InputsMutation::RevKey, 0xFFFFFFFF);

        CudaConfig cudaConfig = CudaConfig::Optimal();

        const auto learningItem = KeeloqLearning::LearningItem(learningSommer);
        auto result = bruteforcer.run(simple, cudaConfig, KeeloqLearning::Matrix({ learningItem }));
        const auto& stats = bruteforcer.getStats();

        if (stats.result != Bruteforcer::Stats::UserCancelled)
        {
            assert(stats.result == Bruteforcer::Stats::Success && "Benchmark real test failed, no successful result");
            assert(result.hasMatch() && "Benchmark real test failed, no match found for Sommer inputs");
            assert(result.match == KeeloqLearning::DecryptedResults::getIndex(learningItem) && "Invalid match for real benchmark test (Sommer)");

            result.print(sommerInputs);
            printf("Real benchmark for Sommer: Time (s): %" PRIu64 "\n\n", timer.elapsedSeconds().count());
        }
    }
}


int benchmark::run(const std::vector<EncParcel>& inputs, const KeeloqLearning::Matrix& learningMatrix,
    const BruteforceConfig& benchmarkConfig, uint32_t numCudaBlocks, uint16_t numCudaThreads)
{
    Bruteforcer bruteforcer(inputs, true, true);
    CudaConfig cudaConfig{ numCudaBlocks, numCudaThreads, 1 };

    bruteforcer.run(benchmarkConfig, cudaConfig, learningMatrix);
    const auto& stats = bruteforcer.getStats();

    console::clearLinesUp(1);

    const bool kernelFailure = stats.result == Bruteforcer::Stats::KernelFailure;
    const bool userCancelled = stats.result == Bruteforcer::Stats::UserCancelled;

    if (kernelFailure || userCancelled)
    {
        const auto resText = kernelFailure ? "FAILURE" : " CANCEL";

        printf("| CUDA: %5" PRIu32 " x %-5" PRIu16 " | MKeys:%4" PRIu64 " | GPU Memory %4.1f GB | Round time: %s | Avg.: %s  | Batch avg. speed: %s  | \n",
            numCudaBlocks, numCudaThreads, stats.totalCalcs() / 1000000, stats.allocatedGB(), resText, resText, resText);

        return kernelFailure ? -1 : 0;
    }

    printf("| CUDA: %5" PRIu32 " x %-5" PRIu16 " | MKeys:%4" PRIu64 " | GPU Memory %4.1f GB | Round time:%6" PRIu64 "ms | Avg.:%6.1f Mk/s | Batch avg. speed:%6.1f Mk/s | \n",
        numCudaBlocks, numCudaThreads,
        stats.totalCalcs() / 1000000,
        stats.allocatedGB(),
        stats.elapsedMs(),
        stats.avgRoundSpeed() / 1000,
        stats.avgBatchSpeed() / 1000);

    return static_cast<int>(stats.avgRoundSpeed());
}

void benchmark::run(const std::vector<EncParcel>& inputs, const KeeloqLearning::Matrix& learningMatrix, const BruteforceConfig& benchmarkConfig)
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
        "Num total calculations             : %" PRIu64 " (Millions)\n"
        "Seed specified                     : %s\n"
        "Inputs mutations:                  : %s\n"
        "Learning                           : %s\n\n",
        MaxCudaBlocks, MaxCudaMemory, MaxCudaThreads,
        (benchmarkConfig.size * benchmarkConfig.getMutations().size() / 1000000),
        (hasSeed ? "true" : "false"),
        benchmarkConfig.mutationsToString().c_str(),
        learningMatrix.toString(&benchmarkConfig).c_str());

    std::vector<Result> results;

    for (uint32_t numThreads = 256; numThreads <= MaxCudaThreads; numThreads *= 2)
    {
        uint32_t maxBlocksForThreads = std::min(CudaConfig::MaxCudaBlocks(numThreads), MaxCudaBlocks);

        for (uint32_t numBlocks = 1024; numBlocks <= maxBlocksForThreads; numBlocks *= 2)
        {
            auto avgSpeed = run(inputs, learningMatrix, benchmarkConfig, numBlocks, numThreads);
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
            Decryptor::Make(0, 1234567, true), InputsMutation::None, TargetCalculationsNumber);

        // EASIEST: Single easiest brute type only
        auto learningMatrix = KeeloqLearning::Matrix{ LearningType::Simple };
        run(inputs, learningMatrix, benchConfigSimpleBrute);

        // HARDEST: Full brute 11 x 2 x 4 permutations
        learningMatrix = KeeloqLearning::Matrix::Everything();
        benchConfigSimpleBrute.setMutationMask(InputsMutation::All);
        benchConfigSimpleBrute.size = TargetCalculationsNumber / 10;

        run(inputs, learningMatrix, benchConfigSimpleBrute);
    }

    {
        auto configXorFixed = BruteforceConfig::GetXorFixBruteforce(Decryptor::Make(0xAABBCCDDEEFFFFFF, 0, true), InputsMutation::RevKey, TargetCalculationsNumber);

        // XORed only
        auto learningMatrix = KeeloqLearning::Matrix::Everything();
        run(inputs, learningMatrix, configXorFixed);
    }

    {
        auto configSeedOnly = BruteforceConfig::GetSeedBruteforce(Decryptor::Make(0xAABBCCDDEEFFFFFF, 0, true), InputsMutation::None, TargetCalculationsNumber);

        // Seed only
        auto learningMatrix = KeeloqLearning::Matrix{ LearningType::Secure };
        run(inputs, learningMatrix, configSeedOnly);
    }


    {
        // Alphabet brute benchmarks
        auto configNoSeed = BruteforceConfig::GetAlphabet(Decryptor::MakeNoSeed(0), InputsMutation::None, "0123456789abcdefgh"_b, TargetCalculationsNumber);
        auto configSeeded = BruteforceConfig::GetAlphabet(Decryptor::Make(0, 1234567, true), InputsMutation::None, "0123456789abcdefgh"_b, TargetCalculationsNumber);

        auto learningMatrix = KeeloqLearning::Matrix{ LearningType::Simple };
        run(inputs, learningMatrix, configSeeded);
        run(inputs, learningMatrix, configNoSeed);

        learningMatrix = KeeloqLearning::Matrix{ LearningType::Secure };
        run(inputs, learningMatrix, configSeeded);

        learningMatrix = KeeloqLearning::Matrix{ LearningType::Faac };
        run(inputs, learningMatrix, configSeeded);

        learningMatrix = KeeloqLearning::Matrix{ LearningType::Simple, LearningType::Normal, LearningType::Xor };
        run(inputs, learningMatrix, configNoSeed);
    }
}
