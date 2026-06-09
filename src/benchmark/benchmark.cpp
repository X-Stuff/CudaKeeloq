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
    bool singleLearningMode;
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
        result.emplace_back(encryptor.click(InputsTransform::None, lType, KeeloqLearning::AlgoType::Normal));
    }
    return result;
}
}

int benchmark::run(const std::vector<EncParcel>& inputs, const BruteforceConfig& benchmarkConfig, uint32_t numCudaBlocks, uint16_t numCudaThreads)
{
    const auto kernelModeStr = benchmarkConfig.useSingleLearningKernels ? "Single" : " Multi";

    Bruteforcer bruteforcer(inputs, true, AppVerbosity::Progress);
    CudaConfig cudaConfig{ numCudaBlocks, numCudaThreads };

    bruteforcer.run(benchmarkConfig, cudaConfig);
    const auto& stats = bruteforcer.getStats();

    console::clearLinesUp(1);

    const bool kernelFailure = stats.result == Bruteforcer::Stats::KernelFailure;
    const bool userCancelled = stats.result == Bruteforcer::Stats::UserCancelled;

    if (kernelFailure || userCancelled)
    {
        console::clearLinesUp(3);

        const auto resText = kernelFailure ? "FAILURE" : " CANCEL";

        printf("| CUDA: %6" PRIu32 " x %-4" PRIu16 " | %s | MKeys:%5" PRIu64 " | Batches: %2" PRIu64 " | GPU: %4.1f GB | Time: %s | Avg: %s    | Batch avg : %s    |\n\n",
            numCudaBlocks, numCudaThreads, kernelModeStr, stats.realProcessedKeys / 1000000, stats.numBatches, stats.allocatedGB(), resText, resText, resText);

        return kernelFailure ? -1 : 0;
    }

    printf("| CUDA: %6" PRIu32 " x %-4" PRIu16 " | %s | MKeys:%5" PRIu64 " | Batches: %2" PRIu64 " | GPU: %4.1f GB | Time:%6" PRIu64 "ms | Avg:%6.1f Mk/s | Batch avg :%6.1f Mk/s |\n\n",
        numCudaBlocks, numCudaThreads, kernelModeStr,
        stats.realProcessedKeys / 1000000, stats.numBatches,
        stats.allocatedGB(),
        stats.elapsedMs(),
        stats.avgRoundSpeed() / 1000,
        stats.avgBatchSpeed() / 1000);

    return static_cast<int>(stats.avgRoundSpeed());
}

void benchmark::run(const std::vector<EncParcel>& inputs, const KeeloqLearning::Matrix& fullMatrix, const BruteforceConfig& benchmarkConfig)
{
    static constexpr float    MinCudaMemoryGB = 0.7f;
    static constexpr double   MinCudaMemory = MinCudaMemoryGB * 1024 * 1024 * 1024;

    static const uint32_t MaxCudaThreads = CudaConfig::MaxCudaThreads();
    static const float    MaxCudaMemoryGB = CudaConfig::MaxGlobalMemoryGB();

    const bool hasSeed = benchmarkConfig.hasSeed();
    const auto reducedMatrix = benchmarkConfig.reduceMatrix(fullMatrix);

    printf("BENCHMARK BEGIN (Esc to skip)\n\n"
        "-----------------------------------------------------------\n");
    printf("Config is : %s\n"
        "-----------------------------------------------------------\n"
        "Min Allocation                     : %.1fGB\n"
        "Max Available GPU Memory           : %.1fGB\n"
        "Max CUDA Threads per block         : %u\n"
        "Total decryptors number            : %" PRIu64 " (Millions)\n"
        "Seed specified                     : %s\n"
        "Inputs transforms:                 : %s\n"
        "Learning                           : %s\n\n",
        BruteforceType::name(benchmarkConfig.type),
        MinCudaMemoryGB, MaxCudaMemoryGB, MaxCudaThreads,
        (benchmarkConfig.size / 1000000),
        (hasSeed ? "true" : "false"),
        benchmarkConfig.transformsToString().c_str(),
        reducedMatrix.toString().c_str());

    std::vector<Result> results;

    for (const auto singleLearning : { false, true })
    {
        const size_t sizeofResult = singleLearning ? sizeof(ThreadResult::Single) : sizeof(ThreadResult::Multi);

        auto ccopy = benchmarkConfig;
        ccopy.setLearningMatrix(reducedMatrix);
        ccopy.useSingleLearningKernels = singleLearning;

        for (uint32_t numThreads = 256; numThreads <= MaxCudaThreads; numThreads *= 2)
        {
            const uint32_t maxBlocksForThreads = CudaConfig::MaxCudaBlocks(numThreads, sizeofResult);

            for (uint32_t numBlocks = 4096; numBlocks <= maxBlocksForThreads; numBlocks *= 2)
            {
                const double runAllocations = numThreads * numBlocks * sizeofResult * 3.0;
                if (runAllocations < MinCudaMemory)
                {
                    // too little allocated memory most likely result won't be great
                    continue;
                }

                auto avgSpeed = run(inputs, ccopy, numBlocks, numThreads);
                if (!avgSpeed)
                {
                    // Cancel request
                    return;
                }

                results.push_back({ numBlocks, numThreads, static_cast<int>(avgSpeed), singleLearning });
            }
        }
    }

    if (results.size() > 0)
    {
        std::sort(results.begin(), results.end(), [](const Result& a, const Result& b) { return a.avgSpeed > b.avgSpeed; });
        auto best = results.front();

        auto alt = std::find_if(results.begin(), results.end(), [&best](const auto& element) { return element.singleLearningMode != best.singleLearningMode; });

        printf("\nBENCHMARK FINISHED:\n");

        printf("\tBest result: %s mode in kernel, %u blocks, %u threads - %.3f million keys/s\n",
            best.singleLearningMode ? "Single learning" : "Multiple learnings",
            best.numBlocks, best.numThreads,
            best.avgSpeed / 1000.0f);

        if (alt != results.end())
        {
            printf("\tAlternative: %s mode in kernel, %u blocks, %u threads - %.3f million keys/s\n",
                alt->singleLearningMode ? "Single learning" : "Multiple learnings",
                alt->numBlocks, alt->numThreads,
                alt->avgSpeed / 1000.0f);
        }

        printf("\n\n");
    }
}

void benchmark::becnhmarkReal(bool useSingleLearningKernels)
{
#if _DEBUG
    static constexpr auto ClearMask = 0xFFFFFFFFF8000000;
#else
    static constexpr auto ClearMask = 0xFFFFFFFF00000000;
#endif

    {
        constexpr uint64_t manDH = 0x8455F43584941223;
        constexpr auto learningDH = KeeloqLearning::LearningType::Simple;

        const std::vector<EncParcel> dhInputs = { 0xFBE31D94BB33DD55, 0xD7577925BB33DD55, 0xAA17DD6CBB33DD55 };

        auto timer = Timer<std::chrono::steady_clock>::start();
        const auto dhStartDecryptor = Decryptor::MakeNoSeed(manDH & ClearMask);

        Bruteforcer bruteforcer(dhInputs, true);
        BruteforceConfig simple = BruteforceConfig::GetBruteforce(dhStartDecryptor, InputsTransform::None, 0xFFFFFFFF);
        simple.setLearningMatrix(KeeloqLearning::Matrix::Everything()); // { learningItem });
        simple.useSingleLearningKernels = useSingleLearningKernels;

        CudaConfig cudaConfig = CudaConfig::Optimal();

        [[maybe_unused]] const auto learningItem = KeeloqLearning::LearningItem(learningDH);
        auto result = bruteforcer.run(simple, cudaConfig);
        const auto& stats = bruteforcer.getStats();

        if (stats.result != Bruteforcer::Stats::UserCancelled)
        {
            assert(stats.result == Bruteforcer::Stats::Success && "Benchmark real test failed, no successful result");
            assert(result.isValid() && "Benchmark real test failed, no match found for DH inputs");
            assert(result.learningType == learningItem.learning && "Invalid LType match for real benchmark test (DH)");
            assert(result.algoType == learningItem.algoType && "Invalid AlgoType match for real benchmark test (DH)");

            result.print();
            printf("Real benchmark for DH: Time (s): %" PRIu64 "\n\n", timer.elapsedSeconds().count());

            assert(result.decryptor.man() == manDH && "Benchmark decryption get the invalid decryptor for DH");
        }
    }

    {
        constexpr uint64_t manSommer = 0x7BCBEED4376EDCBF;
        constexpr auto learningSommer = KeeloqLearning::LearningType::Normal;

        const std::vector<EncParcel> sommerInputs = { 0x54E9888FBB33DD55, 0x10439539BB33DD55, 0x3210C297BB33DD55 };

        auto timer = Timer<std::chrono::steady_clock>::start();
        const auto smStartDecryptor = Decryptor::MakeNoSeed(manSommer & ClearMask);

        Bruteforcer bruteforcer(sommerInputs, true);
        BruteforceConfig simple = BruteforceConfig::GetBruteforce(smStartDecryptor, InputsTransform::RevKey, 0xFFFFFFFF);
        simple.setLearningMatrix(KeeloqLearning::Matrix::Everything()); // { learningItem });
        simple.useSingleLearningKernels = useSingleLearningKernels;

        CudaConfig cudaConfig = CudaConfig::Optimal();

        [[maybe_unused]] const auto learningItem = KeeloqLearning::LearningItem(learningSommer);
        auto result = bruteforcer.run(simple, cudaConfig);
        const auto& stats = bruteforcer.getStats();

        if (stats.result != Bruteforcer::Stats::UserCancelled)
        {
            assert(stats.result == Bruteforcer::Stats::Success && "Benchmark real test failed, no successful result");
            assert(result.isValid() && "Benchmark real test failed, no match found for Sommer inputs");
            assert(result.learningType == learningItem.learning && "Invalid LType match for real benchmark test (Sommer)");
            assert(result.algoType == learningItem.algoType && "Invalid AlgoType match for real benchmark test (Sommer)");

            result.print();
            printf("Real benchmark for Sommer: Time (s): %" PRIu64 "\n\n", timer.elapsedSeconds().count());

            assert(result.decryptor.man() == manSommer && "Benchmark decryption get the invalid decryptor for Sommer");
        }
    }
}

void benchmark::benchmarkSeedAttack(uint32_t TargetCalculationsNumber)
{
    using namespace KeeloqLearning;

    constexpr auto NumInputs = 3;
    auto inputs = makeBenchmarkInputs(0xFF123FF3434FFFFF, NumInputs, LearningType::Serial3);

    auto configSeedOnly = BruteforceConfig::GetSeedBruteforce(Decryptor::Make(0xAABBCCDDEEFFFFFF, 0, true), InputsTransform::None, TargetCalculationsNumber);
    run(inputs, KeeloqLearning::Matrix::Everything(), configSeedOnly);
}

void benchmark::benchmarkNormalAttack(uint32_t TargetCalculationsNumber)
{
    using namespace KeeloqLearning;

    constexpr auto NumInputs = 3;
    auto inputs = makeBenchmarkInputs(0xFF123FF3434FFFFF, NumInputs, LearningType::Serial3);

    // Just using no seed decryptor should turn off all seed-based learning types
    auto configNoSeed = BruteforceConfig::GetBruteforce(Decryptor::MakeNoSeed(0xAABBCCDDEEFFFFFF), InputsTransform::None, TargetCalculationsNumber);
    run(inputs, KeeloqLearning::Matrix::Everything(), configNoSeed);
}

void benchmark::benchmarkXoredAttack(uint32_t TargetCalculationsNumber)
{
    using namespace KeeloqLearning;

    constexpr auto NumInputs = 3;
    auto inputs = makeBenchmarkInputs(0xFF123FF3434FFFFF, NumInputs, LearningType::Serial3);

    // XORed only
    auto configXorFixed = BruteforceConfig::GetXorBruteforce(Decryptor::Make(0xAABBCCDDEEFFFFFF, 0, true), InputsTransform::RevKey, TargetCalculationsNumber / InputTransformVariantsCount);
    run(inputs, KeeloqLearning::Matrix::Everything(), configXorFixed);
}

void benchmark::benchmarkEveryLearningAlone(uint32_t TargetCalculationsNumber)
{
    using namespace KeeloqLearning;

    constexpr auto NumInputs = 3;
    auto inputs = makeBenchmarkInputs(0xFF123FF3434FFFFF, NumInputs, LearningType::Serial3);

    // Alphabet brute benchmarks
    auto alphabetSeeded = BruteforceConfig::GetAlphabet(Decryptor::Make(0, 1234567, true), InputsTransform::None, "0123456789abcdefgh"_b, TargetCalculationsNumber);

    // Simple+1 brute benchmarks
    auto simplePlusOne = BruteforceConfig::GetBruteforce(Decryptor::Make(0, 1234567, true), InputsTransform::None, TargetCalculationsNumber);

    for (auto learningItem : KeeloqLearning::Matrix::Everything().asItems())
    {
        run(inputs, KeeloqLearning::Matrix{ learningItem }, alphabetSeeded);
        run(inputs, KeeloqLearning::Matrix{ learningItem }, simplePlusOne);
    }
}

void benchmark::benchmarkEveryLearningAtOnce(uint32_t TargetCalculationsNumber)
{
    using namespace KeeloqLearning;

    constexpr auto NumInputs = 3;
    auto inputs = makeBenchmarkInputs(0xFF123FF3434FFFFF, NumInputs, LearningType::Serial3);

    // Simple+1 brute benchmarks
    auto benchConfigSimpleBrute = BruteforceConfig::GetBruteforce(Decryptor::Make(0, 1234567, true), InputsTransform::None, TargetCalculationsNumber);
    benchConfigSimpleBrute.setTransforms({ EveryInputTransform::values.begin(), EveryInputTransform::values.end() });
    benchConfigSimpleBrute.size = TargetCalculationsNumber / EveryInputTransform::values.size();

    // HARDEST: Full brute 11 x 4 permutations
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

    benchmarkEveryLearningAtOnce(TargetCalculationsNumber / 10);
    benchmarkEveryLearningAlone(TargetCalculationsNumber);

    benchmarkSeedAttack(TargetCalculationsNumber);
    benchmarkNormalAttack(TargetCalculationsNumber);

    benchmarkXoredAttack(TargetCalculationsNumber);

    becnhmarkReal(true);
    becnhmarkReal(false);

}
