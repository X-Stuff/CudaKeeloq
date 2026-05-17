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

int benchmark::run(const std::vector<EncParcel>& inputs, const KeeloqLearning::Matrix& learningMatrix,
    const BruteforceConfig& benchmarkConfig, uint32_t numCudaBlocks, uint16_t numCudaThreads)
{
    Bruteforcer bruteforcer(inputs, true, AppVerbosity::Progress);
    CudaConfig cudaConfig{ numCudaBlocks, numCudaThreads, 1 };

    bruteforcer.run(benchmarkConfig, cudaConfig, learningMatrix);
    const auto& stats = bruteforcer.getStats();

    console::clearLinesUp(1);

    const bool kernelFailure = stats.result == Bruteforcer::Stats::KernelFailure;
    const bool userCancelled = stats.result == Bruteforcer::Stats::UserCancelled;

    if (kernelFailure || userCancelled)
    {
        const auto resText = kernelFailure ? "FAILURE" : " CANCEL";

        printf("| CUDA: %6" PRIu32 " x %-4" PRIu16 " | MKeys:%5" PRIu64 " | GPU Mem %4.1f GB | Round time: %s | Avg: %s  | Batch avg : %s  |\n\n",
            numCudaBlocks, numCudaThreads, stats.realProcessedKeys / 1000000, stats.allocatedGB(), resText, resText, resText);

        return kernelFailure ? -1 : 0;
    }

    printf("| CUDA: %6" PRIu32 " x %-4" PRIu16 " | MKeys:%5" PRIu64 " | GPU Mem %4.1f GB | Round time:%6" PRIu64 "ms | Avg:%6.1f Mk/s | Batch avg :%6.1f Mk/s |\n\n",
        numCudaBlocks, numCudaThreads,
        stats.realProcessedKeys / 1000000,
        stats.allocatedGB(),
        stats.elapsedMs(),
        stats.avgRoundSpeed() / 1000,
        stats.avgBatchSpeed() / 1000);

    return static_cast<int>(stats.avgRoundSpeed());
}

void benchmark::run(const std::vector<EncParcel>& inputs, const KeeloqLearning::Matrix& fullMatrix, const BruteforceConfig& benchmarkConfig)
{
    static const uint32_t MaxCudaThreads = CudaConfig::MaxCudaThreads();
    static const float    MaxCudaMemory = CudaConfig::MaxGlobalMemoryGB();

    const uint32_t MaxCudaBlocks = benchmarkConfig.useSingleLearningKernels ? std::numeric_limits<uint32_t>::max() : 65536;
    const size_t sizeofResult = benchmarkConfig.useSingleLearningKernels ? sizeof(SingleLearningResult) : sizeof(SingleResult);

    const bool hasSeed = benchmarkConfig.hasSeed();
    const auto reducedMatrix = benchmarkConfig.reduceMatrix(fullMatrix);

    printf("BENCHMARK BEGIN (Esc to skip)\n\nConfig is: %s\n", BruteforceType::name(benchmarkConfig.type));
    printf(
        "Max Available CUDA Blocks          : %u\n"
        "Max Available GPU Memory           : %.1fGB\n"
        "CUDA Threads per block             : %u\n"
        "Total decryptors number            : %" PRIu64 " (Millions)\n"
        "Seed specified                     : %s\n"
        "Inputs mutations:                  : %s\n"
        "Kernel inputs:                     : %s\n"
        "Learning                           : %s\n\n",
        MaxCudaBlocks, MaxCudaMemory, MaxCudaThreads,
        (benchmarkConfig.size / 1000000),
        (hasSeed ? "true" : "false"),
        benchmarkConfig.mutationsToString().c_str(),
        benchmarkConfig.useSingleLearningKernels ? "Single" : "Multiple",
        reducedMatrix.toString().c_str());

    std::vector<Result> results;

    for (uint32_t numThreads = 256; numThreads <= MaxCudaThreads; numThreads *= 2)
    {
        uint32_t maxBlocksForThreads = std::min(CudaConfig::MaxCudaBlocks(numThreads, sizeofResult), MaxCudaBlocks);

        for (uint32_t numBlocks = 4096; numBlocks <= maxBlocksForThreads; numBlocks *= 2)
        {
            auto avgSpeed = run(inputs, reducedMatrix, benchmarkConfig, numBlocks, numThreads);
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

void benchmark::becnhmarkReal(bool useSingleLearningKernels)
{
    {
        constexpr uint64_t manDH = 0x8455F43584941223;
        constexpr auto learningDH = KeeloqLearning::LearningType::Simple;

        const std::vector<EncParcel> dhInputs = { 0xFBE31D94BB33DD55, 0xD7577925BB33DD55, 0xAA17DD6CBB33DD55 };

        auto timer = Timer<std::chrono::steady_clock>::start();
        const auto dhStartDecryptor = Decryptor::MakeNoSeed(manDH & 0xFFFFFFFF00000000);

        Bruteforcer bruteforcer(dhInputs, true);
        BruteforceConfig simple = BruteforceConfig::GetBruteforce(dhStartDecryptor, InputsMutation::None, 0xFFFFFFFF);
        simple.useSingleLearningKernels = useSingleLearningKernels;

        CudaConfig cudaConfig = CudaConfig::Optimal();

        const auto learningItem = KeeloqLearning::LearningItem(learningDH);
        auto result = bruteforcer.run(simple, cudaConfig, KeeloqLearning::Matrix::Everything());// { learningItem });
        const auto& stats = bruteforcer.getStats();

        if (stats.result != Bruteforcer::Stats::UserCancelled)
        {
            assert(stats.result == Bruteforcer::Stats::Success && "Benchmark real test failed, no successful result");
            assert(result.isValid() && "Benchmark real test failed, no match found for DH inputs");
            assert(result.learningType == learningItem.learning && "Invalid LType match for real benchmark test (DH)");
            assert(result.algoModifier == learningItem.amod && "Invalid AMod match for real benchmark test (DH)");

            result.print();
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
        simple.useSingleLearningKernels = useSingleLearningKernels;

        CudaConfig cudaConfig = CudaConfig::Optimal();

        const auto learningItem = KeeloqLearning::LearningItem(learningSommer);
        auto result = bruteforcer.run(simple, cudaConfig, KeeloqLearning::Matrix::Everything());// { learningItem });
        const auto& stats = bruteforcer.getStats();

        if (stats.result != Bruteforcer::Stats::UserCancelled)
        {
            assert(stats.result == Bruteforcer::Stats::Success && "Benchmark real test failed, no successful result");
            assert(result.isValid() && "Benchmark real test failed, no match found for Sommer inputs");
            assert(result.learningType == learningItem.learning && "Invalid LType match for real benchmark test (Sommer)");
            assert(result.algoModifier == learningItem.amod && "Invalid AMod match for real benchmark test (Sommer)");

            result.print();
            printf("Real benchmark for Sommer: Time (s): %" PRIu64 "\n\n", timer.elapsedSeconds().count());
        }
    }
}

void benchmark::benchmarkSeedAttack(uint32_t TargetCalculationsNumber, bool useSingleLearningKernels)
{
    using namespace KeeloqLearning;

    constexpr auto NumInputs = 3;
    auto inputs = makeBenchmarkInputs(0xFF123FF3434FFFFF, NumInputs, LearningType::Serial3);

    auto configSeedOnly = BruteforceConfig::GetSeedBruteforce(Decryptor::Make(0xAABBCCDDEEFFFFFF, 0, true), InputsMutation::None, TargetCalculationsNumber);
    configSeedOnly.useSingleLearningKernels = useSingleLearningKernels;

    run(inputs, KeeloqLearning::Matrix::Everything(), configSeedOnly);
}

void benchmark::benchmarkNormalAttack(uint32_t TargetCalculationsNumber, bool useSingleLearningKernels)
{
    using namespace KeeloqLearning;

    constexpr auto NumInputs = 3;
    auto inputs = makeBenchmarkInputs(0xFF123FF3434FFFFF, NumInputs, LearningType::Serial3);

    // Just using no seed decryptor should turn off all seed-based learning types
    auto configNoSeed = BruteforceConfig::GetBruteforce(Decryptor::MakeNoSeed(0xAABBCCDDEEFFFFFF), InputsMutation::None, TargetCalculationsNumber);
    configNoSeed.useSingleLearningKernels = useSingleLearningKernels;

    run(inputs, KeeloqLearning::Matrix::Everything(), configNoSeed);
}

void benchmark::benchmarkXoredAttack(uint32_t TargetCalculationsNumber, bool useSingleLearningKernels)
{
    using namespace KeeloqLearning;

    constexpr auto NumInputs = 3;
    auto inputs = makeBenchmarkInputs(0xFF123FF3434FFFFF, NumInputs, LearningType::Serial3);

    // XORed only
    auto configXorFixed = BruteforceConfig::GetXorFixBruteforce(Decryptor::Make(0xAABBCCDDEEFFFFFF, 0, true), InputsMutation::RevKey, TargetCalculationsNumber);
    configXorFixed.useSingleLearningKernels = useSingleLearningKernels;

    run(inputs, KeeloqLearning::Matrix::Everything(), configXorFixed);
}

void benchmark::benchmarkEveryLearningAlone(uint32_t TargetCalculationsNumber, bool useSingleLearningKernels)
{
    using namespace KeeloqLearning;

    constexpr auto NumInputs = 3;
    auto inputs = makeBenchmarkInputs(0xFF123FF3434FFFFF, NumInputs, LearningType::Serial3);


    // Alphabet brute benchmarks
    auto alphabetSeeded = BruteforceConfig::GetAlphabet(Decryptor::Make(0, 1234567, true), InputsMutation::None, "0123456789abcdefgh"_b, TargetCalculationsNumber);
    alphabetSeeded.useSingleLearningKernels = useSingleLearningKernels;

    // Simple+1 brute benchmarks
    auto simplePlusOne = BruteforceConfig::GetBruteforce(Decryptor::Make(0, 1234567, true), InputsMutation::None, TargetCalculationsNumber);
    simplePlusOne.useSingleLearningKernels = useSingleLearningKernels;

    for (auto learningItem : KeeloqLearning::Matrix::Everything().asItems())
    {
        run(inputs, KeeloqLearning::Matrix{ learningItem }, alphabetSeeded);
        run(inputs, KeeloqLearning::Matrix{ learningItem }, simplePlusOne);
    }
}

void benchmark::benchmarkEveryLearningAtOnce(uint32_t TargetCalculationsNumber, bool useSingleLearningKernels)
{
    using namespace KeeloqLearning;

    constexpr auto NumInputs = 3;
    auto inputs = makeBenchmarkInputs(0xFF123FF3434FFFFF, NumInputs, LearningType::Serial3);

    // Simple+1 brute benchmarks
    auto benchConfigSimpleBrute = BruteforceConfig::GetBruteforce(Decryptor::Make(0, 1234567, true), InputsMutation::None, TargetCalculationsNumber);
    benchConfigSimpleBrute.useSingleLearningKernels = useSingleLearningKernels;

    // HARDEST: Full brute 11 x 4 permutations
    benchConfigSimpleBrute.overrideMutationMask(InputsMutation::All);
    benchConfigSimpleBrute.size = TargetCalculationsNumber;

    run(inputs, KeeloqLearning::Matrix::Everything(), benchConfigSimpleBrute);
}

void benchmark::all(const CommandLineArgs& /*args*/)
{
#if _DEBUG
    constexpr size_t TargetCalculationsNumber = 10000000;
#else
    constexpr size_t TargetCalculationsNumber = 100000000;
#endif

    console_clear();

    for (bool useSingleLearningKernels : { false, true })
    {
        becnhmarkReal(useSingleLearningKernels);

        benchmarkEveryLearningAtOnce(TargetCalculationsNumber / 10, useSingleLearningKernels);
        benchmarkEveryLearningAlone(TargetCalculationsNumber, useSingleLearningKernels);

        benchmarkSeedAttack(TargetCalculationsNumber, useSingleLearningKernels);
        benchmarkNormalAttack(TargetCalculationsNumber, useSingleLearningKernels);

        benchmarkXoredAttack(TargetCalculationsNumber, useSingleLearningKernels);
    }
}
