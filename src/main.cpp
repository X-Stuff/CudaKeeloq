#include "common.h"

#include "stdio.h"

#include "algorithm/keeloq/keeloq_kernel.h"
#include "bruteforce/generators/generator_bruteforce.h"
#include "bruteforce/bruteforcer.h"

#include "host/console.h"
#include "tests/test_all.h"

CommandLineArgs demoTestCommandlineArgs(int num_gen_input = 3)
{
    constexpr uint64_t debugKey = 0xC0FFEE00DEAD6666;
    constexpr uint32_t debugSeed = 0x12345678;

#if _DEBUG
    uint64_t first = debugKey & 0xFFFFFFFFFFC00000;
#else
    uint64_t first = debugKey & 0xFFFFFFFFF0000000;
#endif
    uint64_t count = 0xFFFFFFF;

    Decryptor first_decryptor_ptrn = Decryptor::Make(0, debugSeed, true);
    Decryptor first_decryptor_brtf = Decryptor::Make(first, debugSeed, true);

    CommandLineArgs cmd;
    cmd.inputs = tests::keeloq::gen_inputs(debugKey, num_gen_input, KeeloqLearning::LearningType::Faac);
    cmd.alphabets.emplace_back(MultibaseDigit("abcdef"_b));
    cmd.alphabets.emplace_back(MultibaseDigit( { 0xC0, 0xFF, 0xEE, 0x00, 0xDE, 0xAD, 0x66 }));

    // Dictionary
    cmd.brute_configs.emplace_back(BruteforceConfig::GetDictionary({
        Decryptor::Make(666, debugSeed, true),
        Decryptor::Make(debugKey - 1, debugSeed, true),
        Decryptor::Make(debugKey, debugSeed, true),
        Decryptor::Make(debugKey + 1, debugSeed, true)
    }));

    // Alphabet
    cmd.brute_configs.emplace_back(BruteforceConfig::GetAlphabet(first_decryptor_ptrn, cmd.alphabets[1]));
    cmd.brute_configs.emplace_back(BruteforceConfig::GetAlphabet(first_decryptor_ptrn, cmd.alphabets[0]));

    // Seed
    cmd.brute_configs.emplace_back(BruteforceConfig::GetSeedBruteforce(Decryptor::Make(debugKey, 0, true)));

    // Pattern (reversed)
    cmd.brute_configs.emplace_back(BruteforceConfig::GetPattern(first_decryptor_ptrn, BruteforcePattern(
        {
            BruteforcePattern::ParseBytes("c0|c1|c2|c3"),
            BruteforcePattern::ParseBytes("F0-FF"),
            BruteforcePattern::ParseBytes("E0-EF"),
            BruteforcePattern::ParseBytes("00-99"),
            { 0xED, 0xDE },
            { 0xDA, 0xAD },
            { 0x11, 0x22, 0x33, 0x44, 0x55, 0x66 },
            { 0x11, 0x22, 0x33, 0x44, 0x55, 0x66 }
        }, "N/A")));

    // Simple
    cmd.brute_configs.emplace_back(BruteforceConfig::GetBruteforce(first_decryptor_brtf, count));

    // Filters
    cmd.brute_configs.emplace_back(BruteforceConfig::GetBruteforce(first_decryptor_brtf, count, BruteforceFilters
        {
            // Include only
            BruteforceFilters::Flags::All,

            // Exclude
            BruteforceFilters::Flags::BytesIncremental
        }));

    cmd.selected_learning = { }; // ALL

    cmd.match_stop = false;
    cmd.run_bench = false;
    cmd.run_tests = false;
    cmd.init_cuda();

    return cmd;
}

void bruteforce(const CommandLineArgs& args)
{
    if (args.selected_learning.size() == 0)
    {
        printf("Bruteforcing without specific learning type (slower)"
            "(1 KKey/s == %u Kkc (keeloq calcs) per second)\n"
            "In case of full range there also redundant checks since using _REV learning types ( X-00:11:22 == X_REV-22:11:00 )\n", KeeloqLearning::DecryptedArraySize);
    }

    printf("Total bruteforce configs to run: %zd\n", args.brute_configs.size());

    Bruteforcer bruteforcer(args.inputs);

    for (const auto& config : args.brute_configs)
    {
        printf("--------------------------------------------------------------------------------------------------------------");
        auto learningMatrix = KeeloqLearning::Matrix(args.selected_learning, args.selected_input_mods, args.selected_algo_mods);

        SingleResult result = bruteforcer.run(config, args.cudaConfig(), learningMatrix);
        if (result.hasMatch())
        {
            result.print(args.inputs);

            if (args.match_stop)
            {
                break;
            }
        }
    }
}

int main(int argc, const char** argv)
{
    assert(tests::cuda_check_working());

    if (!keeloq::kernels::cuda_is_working())
    {
        printf("Error: This device cannot compute keeloq right. Single encryption and decryption mismatch.\n");
        assert(false);
        return 1;
    }

    if (KeeloqLearning::DecryptedResults::cuda_init() != cudaSuccess)
    {
        printf("Error: Failed to initialize constants for DecryptedResults cache on device.\n");
        assert(false);
        return 1;
    }

    // Be default if no arguments specified - launch demo mode
    bool demo_mode = argc <= 1;
    auto args = demo_mode ? demoTestCommandlineArgs() : CommandLineArgs::parse(argc, argv);
    if (args.print_version)
    {
        printf("Version: " APP_VERSION_STRING "\n");
        return 0;
    }

    bool had_tests = args.run_bench || args.run_tests;

    if (args.run_tests)
    {
        printf("\n...RUNNING TESTS...\n");
#if _DEBUG
        tests::console::run();
#endif

        bool tests_ok = tests::check_utils();

        tests_ok &= tests::generators::all();
        tests_ok &= tests::alphabet_generation();
        tests_ok &= tests::filters_generation();
        tests_ok &= tests::keeloq::all();

        if (!tests_ok)
        {
            printf("\n...TESTS FAILED...\n");
            return 1;
        }
        printf("\n...TESTS FINISHED...\n");
    }

    if (args.run_bench)
    {
        benchmark::all(args);
    }

    if (args.can_bruteforce())
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
    else if (!had_tests)
    {
        printf("\nNot enough arguments for bruteforce\n");
        return 1;
    }

    // this will free all memory as well
    cudaDeviceReset();

    console::set_cursor_state(true);
    return 0;
}