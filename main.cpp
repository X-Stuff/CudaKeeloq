#include "common.h"

#include <numeric>
#include <future>
#include <thread>

#include "stdio.h"

#include "host/console.h"

#include "algorithm/keeloq/keeloq_kernel.h"

#include "bruteforce/bruteforce_round.h"
#include "bruteforce/generators/generator_bruteforce.h"

#include "tests/test_all.h"


void benchmark(const CommandLineArgs& args)
{
#if _DEBUG
    constexpr size_t TargetCalculations = 10000000;
#else
    constexpr size_t TargetCalculations = 100000000;
#endif

    constexpr uint16_t CudaBlocks[] =  { 128, 256, 512, 1024, 2048, 4096, 8196 };
    constexpr uint16_t CudaThreads[] = { 128, 256, 512, 1024, 2048 };

    static const uint32_t MaxCudaThreads = CommandLineArgs::max_cuda_threads();
    static const uint32_t MaxCudaBlocks  = CommandLineArgs::max_cuda_blocks();

    BruteforceConfig benchmarkConfig = BruteforceConfig::GetAlphabet(0, "0123456789abcdefgh"_b);

    printf("BENCHMARK BEGIN\n\n"
        "Config is: Alphabet\n"
        "Num loops inside CUDA\t\t\t: %u (from cmd)\n"
        "Max Available CUDA Threads per block\t: %u\n"
        "Max Available CUDA Blocks\t\t: %u\n"
        "Num total calculations\t\t\t: %llu (Millions)\n"
        "Learning (from cmd)\t\t\t: %s\n\n",
        args.cuda_loops,
        MaxCudaThreads, MaxCudaBlocks,
        (TargetCalculations / 1000000),
        KeeloqLearningType::to_string(args.selected_learning).c_str());


    bool in_progress = true;
    for (auto& NumCudaBlocks : CudaBlocks)
    {
        if (!in_progress || NumCudaBlocks > MaxCudaBlocks)
        {
            break;
        }

        for (auto& NumCudaThreads : CudaThreads)
        {
            if (!in_progress || NumCudaThreads > MaxCudaThreads)
            {
                break;
            }

            BruteforceRound benchmarkRound(args.inputs, benchmarkConfig, args.selected_learning, NumCudaBlocks, NumCudaThreads, args.cuda_loops);
            benchmarkRound.Init();

            size_t keysInBatch	= benchmarkRound.keys_per_batch();
            size_t numBatches = std::max(1ull, TargetCalculations / keysInBatch);

            auto roundStartTime = std::chrono::system_clock::now();

            std::vector<uint64_t> batches_kResults_per_sec(numBatches);

            console_hide_cursor();
            printf("\n");

            for (int i = 0; in_progress && i < numBatches; ++i)
            {
                auto batchStartTime = std::chrono::high_resolution_clock::now();

                KeeloqKernelInput& kernelInput = benchmarkRound.Inputs();
                kernelInput.NextDecryptor();

                GeneratorBruteforce::PrepareDecryptors(kernelInput, NumCudaBlocks, NumCudaThreads);
                LaunchKeeloqBruteMain(kernelInput, NumCudaBlocks, NumCudaThreads);

                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::high_resolution_clock::now() - batchStartTime);

                batches_kResults_per_sec[i] = duration.count() == 0 ? 0 : keysInBatch / duration.count();

                auto overall = std::chrono::duration_cast<std::chrono::seconds>(
                    std::chrono::system_clock::now() - roundStartTime);

                console_cursor_ret_up(1);
                console::progress_bar( i / (double)numBatches, overall);

                if (console::read_esc_press())
                {
                    console_cursor_ret_up(1);
                    console::clear_line();
                    printf("Benchmark skipped\n");
                    in_progress = false;
                }
            }

            if (in_progress)
            {
                auto overall = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::system_clock::now() - roundStartTime);

                std::sort(batches_kResults_per_sec.begin(), batches_kResults_per_sec.end());

                console_cursor_ret_up(1);
                console::clear_line();

                // Creating results
                printf(str::format<std::string>(
                    "| CUDA: %u x %u \t| MEM: %u MB\t | Time (ms): %llu\t |\tSpeed (K/s): %lu (avg.) %lu (mean) |\t\t\t\t\n",
                    NumCudaBlocks, NumCudaThreads,
                    benchmarkRound.get_mem_size() / (1024 * 1024),
                    overall.count(),
                    std::reduce(batches_kResults_per_sec.begin(), batches_kResults_per_sec.end()) / numBatches,
                    batches_kResults_per_sec[numBatches / 2]
                    ).c_str());
            }
        }
    }
}

void benchmark_all(const CommandLineArgs& args)
{
    console_clear();
    CommandLineArgs copy = args;

    copy.cuda_loops = 2;
    copy.selected_learning = {};
    benchmark(copy);

    copy.cuda_loops = 2;
    copy.selected_learning = { KeeloqLearningType::Simple };
    benchmark(copy);

    copy.cuda_loops = 4;
    benchmark(copy);
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
                assert(error == 0);
            }
            else
            {
                // Write next batch of keys from dictionary
                kernelInput.WriteDecryptors(config.decryptors, batch * keysInBatch, keysInBatch);
            }

            // do the bruteforce
            auto kernelResults = LaunchKeeloqBruteMain(kernelInput, attackRound.CudaBlocks(), attackRound.CudaThreads());
            match = attackRound.check_results(kernelResults);

            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - batchStartTime);

            if (batch == 0 || match)
            {
                console_hide_cursor();
                printf("\n\n\n");
            }

            if (!match)
            {
                auto kilo_result_per_second = duration.count() == 0 ? 0 : keysInBatch / duration.count();
                auto progress_percent = (double)(batch + 1) / batchesInRound;

                console_cursor_ret_up(2);

                printf("[%c][%zd/%zd]\t %llu(ms)/batch Speed: %llu KKeys/s\tLast key:0x%llX (%ul)\n", WAIT_CHAR(batch),
                    batch, batchesInRound, duration.count(),
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

    if (!CUDA_check_keeloq_works())
    {
        printf("Error: This device cannot compute keeloq right. Single encryption and decryption mismatch.");
        assert(false);
        return 1;
    }


    const char* commandline[] = {
        "tests",
        "--" ARG_INPUTS"=0xC65D52A0A81FD504,0xCCA9B335A81FD504,0xE0DA7372A81FD504",
        "--" ARG_MODE"=0,1,2,4,3",
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

        "--" ARG_TEST"=1",
        //"--" ARG_BENCHMARK"=1",
    };

    console_set_width(CONSOLE_WIDTH);


    auto args = argc > 1 ? console::parse_command_line(argc, argv) :
        console::parse_command_line(sizeof(commandline) / sizeof(char*), commandline); //console::parse_command_line(argc, argv);
    if (args.run_bench)
    {
        benchmark_all(args);
        return 0;
    }

    if (args.run_tests)
    {
        printf("\n...RUNNING TESTS...\n");
        console::tests::run();

        Tests::AlphabetGeneration();
        Tests::FiltersGeneration();

        printf("\n...TESTS FINISHED...\n");

        //return 0;
        console_clear();
    }

    if (args.can_bruteforce())
    {
        bruteforce(args);
    }
    else
    {
        printf("\nInvalid arguments specified.");
        return 1;
    }


    // this will free all memory as well
    cudaDeviceReset();
    return 0;
}