#include "common.h"

#include "stdio.h"

#include "host/console.h"

#include "algorithm/keeloq/keeloq_kernel.h"
#include "bruteforce/generators/generator_bruteforce.h"
#include "bruteforce/bruteforce_round.h"

#include "tests/test_all.h"


constexpr const char* debugTestCommandline[] = {
    "tests",
    "--" ARG_INPUTS"=0xC65D52A0A81FD504,0xCCA9B335A81FD504,0xE0DA7372A81FD504",
    "--" ARG_MODE"=4,3,0,1,2",
    //"--" ARG_LTYPE"=6,1,3",

    "--" ARG_WORDDICT"=0xCEB6AE48B5C63ED1,0xCEB6AE48B5C63ED2,0xCEB6AE48B5C63ED3",

#if _DEBUG
    "--" ARG_BLOCKS"=512",
    "--" ARG_LOOPS"=2",
    //"--" ARG_START"=0xCEB6AE48B0000000",
#else
    "--" ARG_BLOCKS"=8196",
    "--" ARG_LOOPS"=2",
    "--" ARG_START"=0xCEB6AE4800000000",
#endif
    "--" ARG_COUNT"=0xFFFFFFFF",

    // "--" ARG_IFILTER"=0x2", include filter let be all (otherwise will have big impact)
    "--" ARG_EFILTER"=96",  // BytesRepeat4 | BytesIncremental should increase performance(?)

    "--" ARG_ALPHABET"=CE:B6:AE:48:B5:C6:3E:D2:AA:BB:CC:DD,examples/alphabet.bin,CE:B6:AE:48:B5:C6:3E:D2",//:AA:BB:CC:DD:EE:FF:00:11",

    "--" ARG_PATTERN"=*:B6:AE:*:B5:C0-CF:1E|2E|3E|4E|5E|6E:D2",

    "--" ARG_FMATCH"=0",

    "--" ARG_TEST"=true",
    //"--" ARG_BENCHMARK"=1",
};



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

        printf("\rRunning...\t\t\t\n%s\n", attackRound.to_string().c_str());

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
                if (batch > 0)
                {
                    // Make previous last generated key be an initial for current generation batch
                    kernelInput.NextDecryptor();
                }

                // Generate decryptors (if available)
                int error = GeneratorBruteforce::PrepareDecryptors(kernelInput, attackRound.CudaBlocks(), attackRound.CudaThreads());
                if (error)
                {
                    printf("Error: Key generation resulted with error: %d", error);
                    assert(false);
                    return;
                }
            }
            else
            {
                // Write next batch of keys from dictionary
                kernelInput.WriteDecryptors(config.decryptors, batch * keysInBatch, keysInBatch);
            }

            // do the bruteforce
            auto kernelResults = keeloq::kernels::BruteMain(kernelInput, attackRound.CudaBlocks(), attackRound.CudaThreads());
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
                    kernelInput.config.last.man, kernelInput.config.last.seed);

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
    assert(Tests::CheckCudaIsWorking());

    if (!keeloq::kernels::IsWorking())
    {
        printf("Error: This device cannot compute keeloq right. Single encryption and decryption mismatch.");
        assert(false);
        return 1;
    }

    console_set_width(CONSOLE_WIDTH);

    auto args = argc > 1 ? console::parse_command_line(argc, argv) :
        console::parse_command_line(sizeof(debugTestCommandline) / sizeof(char*), (const char**)debugTestCommandline);

    bool should_bruteforce = args.run_bench || args.run_tests;

    if (should_bruteforce)
    {
        if (args.run_tests)
        {
            printf("\n...RUNNING TESTS...\n");
            console::tests::run();

            Tests::PatternGeneration();
            Tests::AlphabetGeneration();
            Tests::FiltersGeneration();

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