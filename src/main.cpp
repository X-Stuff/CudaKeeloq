#include <csignal>
#include <cstdio>
#include <cstdlib>

#include "common.h"

#include "algorithm/keeloq/keeloq_kernel.h"
#include "algorithm/keeloq/keeloq_encryptor.h"

#include "benchmark/benchmark.h"

#include "bruteforce/bruteforcer.h"
#include "bruteforce/generators/generator_bruteforce.h"

#include "host/command_line_args.h"
#include "host/console.h"


CommandLineArgs demoTestCommandlineArgs(int num_gen_input = 3)
{
    constexpr uint64_t EncryptorKey = 0xC0FFEE00DEAD6666;
    constexpr uint32_t EncryptorSeed = 0x12345678;

    Encryptor encryptor(EncryptorKey, EncryptorSeed);


#if _DEBUG
    uint64_t first = encryptor.getKey() & 0xFFFFFFFFFFC00000;
#else
    uint64_t first = encryptor.getKey() & 0xFFFFFFFFF0000000;
#endif
    uint64_t count = 0xFFFFFFF;

    Decryptor first_decryptor_ptrn = Decryptor::Make(0, encryptor.getSeed(), true);

    CommandLineArgs cmd;
    cmd.inputs.emplace_back(encryptor.click(InputsTransform::None, KeeloqLearning::LearningType::Faac, KeeloqLearning::Modifier::Algo::Normal));
    cmd.inputs.emplace_back(encryptor.click(InputsTransform::None, KeeloqLearning::LearningType::Faac, KeeloqLearning::Modifier::Algo::Normal));
    cmd.inputs.emplace_back(encryptor.click(InputsTransform::None, KeeloqLearning::LearningType::Faac, KeeloqLearning::Modifier::Algo::Normal));

    // Dictionary
    cmd.brute_configs.emplace_back(BruteforceConfig::GetDictionary({
        Decryptor::Make(666, 777, true),
        Decryptor::Make(encryptor.getKey() - 1, encryptor.getSeed(), true),
        Decryptor::Make(encryptor.getKey(),     encryptor.getSeed(), true),
        Decryptor::Make(encryptor.getKey() + 1, encryptor.getSeed(), true)
    }, InputsTransform::None));

    // Alphabet
    cmd.alphabets.emplace_back(MultibaseDigit("abcdef"_b));
    cmd.alphabets.emplace_back(MultibaseDigit({ 0xC0, 0xFF, 0xEE, 0x00, 0xDE, 0xAD, 0x66 }));
    cmd.brute_configs.emplace_back(BruteforceConfig::GetAlphabet(first_decryptor_ptrn, InputsTransform::None, cmd.alphabets[1], BruteforceConfig::MaxDecryptorsNum, "Match alphabet"));
    cmd.brute_configs.emplace_back(BruteforceConfig::GetAlphabet(first_decryptor_ptrn, InputsTransform::None, cmd.alphabets[0], BruteforceConfig::MaxDecryptorsNum, "Wrong alphabet"));

    // Seed
    cmd.brute_configs.emplace_back(BruteforceConfig::GetSeedBruteforce(Decryptor::Make(encryptor.getKey(), 0, true), InputsTransform::None));

    // Pattern (reversed)
    cmd.brute_configs.emplace_back(BruteforceConfig::GetPattern(first_decryptor_ptrn, InputsTransform::RevKey, BruteforcePattern(
        {
            BruteforcePattern::parseBytes("c0|c1|c2|c3"),
            BruteforcePattern::parseBytes("F0-FF"),
            BruteforcePattern::parseBytes("E0-EF"),
            BruteforcePattern::parseBytes("00-99"),
            { 0xED, 0xDE },
            { 0xDA, 0xAD },
            { 0x11, 0x22, 0x33, 0x44, 0x55, 0x66 },
            { 0x11, 0x22, 0x33, 0x44, 0x55, 0x66 }
        }, "Match custom pattern")));

    // Simple
    Decryptor first_decryptor_brtf = Decryptor::Make(first, encryptor.getSeed(), true);
    cmd.brute_configs.emplace_back(BruteforceConfig::GetBruteforce(first_decryptor_brtf, InputsTransform::None, count));

    // Filters
    cmd.brute_configs.emplace_back(BruteforceConfig::GetBruteforce(first_decryptor_brtf, InputsTransform::None, count, BruteforceFilters
        {
            // Include only
            BruteforceFilters::Flags::All,

            // Exclude
            BruteforceFilters::Flags::BytesIncremental
        }));

    cmd.match_stop = false;
    cmd.run_bench  = false;

    return cmd;
}

void bruteforce(const CommandLineArgs& args)
{
    const size_t numConfigs = args.brute_configs.size();
    printf("\nTotal bruteforce configs to run: %zd\n", numConfigs);

    Bruteforcer bruteforcer(args.inputs);

    for (size_t configIndex = 0; configIndex < numConfigs; ++configIndex)
    {
        printf("\n*********************************************[CONFIG %02zd/%02zd]********************************************\n", configIndex + 1, numConfigs);

        const auto& config = args.brute_configs[configIndex];
        const auto cudaConfig = config.cudaConfig(args.cudaBlocks(), args.cudaThreads());

        auto match = bruteforcer.run(config, cudaConfig);
        if (match.isValid())
        {
            match.print();

            if (args.match_stop)
            {
                break;
            }
        }
    }
}

static void restoreCursor()
{
    console::setCursorState(true);
}

static void signalRestoreCursor(int sig)
{
    restoreCursor();
    std::signal(sig, SIG_DFL);
    std::raise(sig);
}

int main(int argc, const char** argv)
{
    std::atexit(restoreCursor);
    std::signal(SIGINT,  signalRestoreCursor);
    std::signal(SIGTERM, signalRestoreCursor);
    std::signal(SIGABRT, signalRestoreCursor);
    std::signal(SIGSEGV, signalRestoreCursor);
    std::signal(SIGILL,  signalRestoreCursor);
    std::signal(SIGFPE,  signalRestoreCursor);

    if (!keeloq::kernels::cuda_is_working())
    {
        printf("Error: This device cannot compute keeloq right. Single encryption and decryption mismatch.\n");
        return 1;
    }

    if (KeeloqLearning::DecryptedResults::cuda_init() != cudaSuccess)
    {
        printf("Error: Failed to initialize constants for DecryptedResults cache on device.\n");
        return 1;
    }

    // Be default if no arguments specified - launch demo mode
    bool demo_mode = argc <= 1;
    auto args = demo_mode ? demoTestCommandlineArgs() : CommandLineArgs::parse(argc, argv);

    if (args.print_help)
    {
        CommandLineArgs::printHelp();
        return 0;
    }

    if (args.print_version)
    {
        printf("Version: " APP_VERSION_STRING "\n");
        return 0;
    }

    if (args.run_bench)
    {
        benchmark::all(args);
    }

    if (args.has_errors)
    {
        return 1;
    }

    if (args.canBruteforce())
    {
        if (demo_mode)
        {
            printf(R"(
                     ___                   __  ___        __
                    / _ \___ __ _  ___    /  |/  /__  ___/ /__
                   / // / -_)  ' \/ _ \  / /|_/ / _ \/ _  / -_)
                  /____/\__/_/_/_/\___/ /_/  /_/\___/\_,_/\__/

                )" "\nPrepared inputs, trying to bruteforce with different modes.\n");
        }

        bruteforce(args);
    }
    else if (!args.run_bench)
    {
        printf("\nNot enough arguments for bruteforce\n");
        return 1;
    }

    // this will free all memory as well
    cudaDeviceReset();

    return 0;
}
