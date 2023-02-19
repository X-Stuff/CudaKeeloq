#include "common.h"

#include "stdio.h"

#include "host/console.h"

#include "algorithm/keeloq/keeloq_kernel.h"
#include "bruteforce/generators/generator_bruteforce.h"
#include "bruteforce/bruteforce_round.h"

#include "tests/test_all.h"


CommandLineArgs debugTestCommandlineArgs(int num_gen_input = 3)
{
    constexpr uint64_t debugKey =  0xC0FFEE00DEAD6666;

#if _DEBUG
    uint64_t first = debugKey & 0xFFFFFFFFFF000000;
#else
    uint64_t first = debugKey & 0xFFFFFFFFF0000000;
#endif
    uint64_t count = 0xFFFFFFF;

    Decryptor first_decryptor_ptrn(0, tests::keeloq::default_seed);
    Decryptor first_decryptor_brtf(first, tests::keeloq::default_seed);


    CommandLineArgs cmd;
    cmd.inputs = tests::keeloq::gen_inputs<KeeloqLearningType::Faac>(debugKey, num_gen_input);
    cmd.alphabets.emplace_back(MultibaseDigit("abcdef"_b));
    cmd.alphabets.emplace_back(MultibaseDigit( { 0xC0, 0xFF, 0xEE, 0x00, 0xDE, 0xAD, 0x66 }));

    // Dictionary
    cmd.brute_configs.emplace_back(BruteforceConfig::GetDictionary({
        Decryptor(666, tests::keeloq::default_seed),
        Decryptor(debugKey - 1, tests::keeloq::default_seed),
        Decryptor(debugKey, tests::keeloq::default_seed),
        Decryptor(debugKey + 1, tests::keeloq::default_seed)
    }));

    // Alphabet
    cmd.brute_configs.emplace_back(BruteforceConfig::GetAlphabet(first_decryptor_ptrn, cmd.alphabets[1]));
    cmd.brute_configs.emplace_back(BruteforceConfig::GetAlphabet(first_decryptor_ptrn, cmd.alphabets[0]));

    // Pattern
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
    cmd.run_bench = true;
    cmd.run_tests = false;

#if _DEBUG
    cmd.init_cuda(512, 0, 1);
#else
    cmd.init_cuda(8196, 0, 1);
#endif

    return cmd;
}

void bruteforce(const CommandLineArgs& args)
{
    if (args.selected_learning.size() == 0)
    {
        printf("Bruteforcing without specific learning type (slower)"
            "(1 KKey/s == %u Kkc (keeloq calcs) per second)\n"
            "In case of full range there also redundant checks since using _REV learning types ( X-00:11:22 == X_REV-22:11:00 )\n", KeeloqLearningType::LAST);
    }

    for (const auto& config : args.brute_configs)
    {
        BruteforceRound attackRound(args.inputs, config, args.selected_learning, args.cuda_blocks, args.cuda_threads, args.cuda_loops);

        printf("\nallocating...");
        attackRound.Init();

        printf("\rRunning...    \n%s\n", attackRound.to_string().c_str());

        bool match = false;

        size_t batchesInRound = attackRound.num_batches();
        size_t keysInBatch = attackRound.keys_per_batch();

        auto roundStartTime = std::chrono::system_clock::now();

        for (size_t batch = 0; !match && batch < batchesInRound; ++batch)
        {
            auto batchStartTime = std::chrono::high_resolution_clock::now();

            KeeloqKernelInput& kernelInput = attackRound.Inputs();

            if (attackRound.Type() != BruteforceType::Dictionary)
            {
                // Generate decryptors (if available)
                int error = GeneratorBruteforce::PrepareDecryptors(kernelInput, attackRound.CudaBlocks(), attackRound.CudaThreads());
                if (error)
                {
                    printf("Error: Key generation resulted with error: %d", error);
                    assert(false);
                    return;
                }

                // Make previous last generated key be an initial for current generation batch
                kernelInput.NextDecryptor();
            }
            else
            {
                // Write next batch of keys from dictionary
                kernelInput.WriteDecryptors(config.decryptors, batch * keysInBatch, keysInBatch);
            }

            // do the bruteforce
            auto kernelResults = keeloq::kernels::cuda_brute(kernelInput, attackRound.CudaBlocks(), attackRound.CudaThreads());
            match = attackRound.check_results(kernelResults);

            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - batchStartTime);

            if (batch == 0 || match)
            {
                console::set_cursor_state(false);
                printf("\n\n\n");
            }

            if (!match)
            {
                auto kilo_result_per_second = duration.count() == 0 ? 0 : keysInBatch / duration.count();
                auto progress_percent = (double)(batch + 1) / batchesInRound;

                console_cursor_ret_up(2);

                printf("[%c][%zd/%zd]\t %" PRIu64 "(ms)/batch Speed: %" PRIu64 " KKeys/s\tLast key:0x%" PRIX64 " (%u)\n",
                    WAIT_CHAR(batch),
                    batch, batchesInRound,
                    duration.count(),
                    kilo_result_per_second,
                    kernelInput.config.last.man(), kernelInput.config.last.seed());

                auto overall = std::chrono::duration_cast<std::chrono::seconds>(
                    std::chrono::system_clock::now() - roundStartTime);

                console::progress_bar(progress_percent, overall);
            }
        }

        if (!match)
        {
            printf("\n\nAfter: %zd batches no results was found. Keys checked:%zd\n\n",
                batchesInRound, batchesInRound * keysInBatch);
        }
        else if (args.match_stop)
        {
            break;
        }
    }
}

int main(int argc, const char** argv)
{
    assert(tests::cuda_check_working());

    if (!keeloq::kernels::cuda_is_working())
    {
        printf("Error: This device cannot compute keeloq right. Single encryption and decryption mismatch.");
        assert(false);
        return 1;
    }

    console_set_width(CONSOLE_WIDTH);

    auto args = argc > 1 ? console::parse_command_line(argc, argv) : debugTestCommandlineArgs();

    bool should_bruteforce = args.run_bench || args.run_tests;

    if (should_bruteforce)
    {
        if (args.run_tests)
        {
            printf("\n...RUNNING TESTS...\n");
            tests::console::run();

            tests::pattern_generation();
            tests::alphabet_generation();
            tests::filters_generation();

            printf("\n...TESTS FINISHED...\n");
        }

        if (args.run_bench)
        {
            benchmark::all(args);
        }
    }
    else
    {
        if (args.can_bruteforce())
        {
            bruteforce(args);
        }
        else
        {
            printf("\nNot enough arguments for bruteforce\n");
            return 1;
        }

        // this will free all memory as well
        cudaDeviceReset();
    }

    console::set_cursor_state(true);
    return 0;
}