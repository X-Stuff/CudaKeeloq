#include <vector>
#include <chrono>
#include <string>
#include <stdlib.h>

#include "console.cuh"
#include "kernel.cuh"
#include "keeloq_main.cuh"
#include "keeloq_generators.cuh"

#include "CUDA_helpers.cuh"


bool process_block_results(const KernelResult& result, CudaRunSetup& run)
{
    if (result.error != 0)
    {
        if (result.error < 0)
        {
            printf("Kernel fatal error: %d\n", result.error);
            return true;
        }
        else
        {
            printf("CUDA calculations num errors: %d\n", result.error);
        }
    }

    if (result.value > 0)
    {
        auto& all_results = run.ReadResults();

        printf("Num matches: %d\n", result.value);

        for (const auto& result : all_results)
        {
            if (result.match == KeeloqLearningType::INVALID)
            {
                continue;
            }

            result.print();
        }

        return true;
    }

    return false;
}

void bruteforce(const CommandLineArgs& args)
{
    if (args.selected_learning == KeeloqLearningType::INVALID)
    {
        printf("Bruteforcing without specific learning type (slower)"
            "(1 KKey/s == %u Kkc (keeloq calcs) per second)\n", KeeloqLearningType::LAST);
    }

    for (const auto& config: args.brute_configs)
    {
        CudaRunSetup single_run(args.inputs, config, args.cuda_blocks, args.cuda_threads, args.cuda_loops);

        printf("\nallocating...");
        single_run.Init();

        printf("\rRunning %s", single_run.ToString().c_str());

        bool match = false;

        size_t num_batches = single_run.NumBatches();
        size_t key_per_batch = single_run.KeysCheckedInBatch();

        auto dec_start = std::chrono::system_clock::now();

        for (size_t batch = 0; !match && batch < num_batches; ++batch)
        {
            auto batch_start = std::chrono::high_resolution_clock::now();

            KernelInput& kernel_input = single_run.Inputs();

            if (single_run.Type() == BruteforceConfig::Type::Dictionary) {
                kernel_input.WriteDecryptors(config.decryptors, batch * key_per_batch, key_per_batch);
            }
            else {
                kernel_input.UpdateInitialDecryptor();
            }

            int error = CUDA_generator_wrapper(kernel_input, args.cuda_blocks, args.cuda_threads);
            assert(error == 0);
            auto kernel_result = CUDA_keeloq_main_wrapper(kernel_input, args.cuda_blocks, args.cuda_threads);
            match = process_block_results(kernel_result, single_run);

            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - batch_start);

            if (batch == 0 || match)
            {
                printf("\n\n");
                save_cursor_pos();
            }

            if (!match)
            {
                auto kilo_result_per_second = duration.count() == 0 ? 0 : key_per_batch / duration.count();
                auto progress_percent = (double)(batch + 1)/ num_batches;

                load_cursor_pos();
                printf("[%c][%zd/%zd]\t %llu(ms)/batch Speed: %llu KKeys/s\tNext key:0x%llX (%ul)\n", WAIT_CHAR(batch),
                    batch, num_batches, duration.count(),
                    kilo_result_per_second,
                    kernel_input.generator.next.man, kernel_input.generator.next.seed);

                auto overall = std::chrono::duration_cast<std::chrono::seconds>(
                    std::chrono::system_clock::now() - dec_start);

                console::progress_bar(progress_percent, overall);
            }
        }

        if (!match)
        {
            printf("\n\nAfter: %zd batches no results was found. Keys checked:%zd\n\n",
                num_batches, num_batches * key_per_batch);
        }
        else if (args.match_stop)
        {
            break;
        }
    }
}

int main(int argc, const char** argv)
{
    //console::tests::run(); return;

    const char* commandline[] = {
        "tests",
        "--" ARG_INPUTS"=0xC65D52A0A81FD504,0xCCA9B335A81FD504,0xE0DA7372A81FD504",
        "--" ARG_BLOCKS"=32",
        "--" ARG_THREADS"=256",
        "--" ARG_LOOPS"=512",
        "--" ARG_MODE"=1,0",

        "--" ARG_WORDDICT"=0xCEB6AE48B5C63ED1,0xCEB6AE48B5C63ED2,0xCEB6AE48B5C63ED3",

        "--" ARG_START"=0xCEB6AE48B0000000",
        "--" ARG_COUNT"=0xFFFFFFF",

        "--" ARG_IFILTER"=0x2", //SmartFilterFlags::Max6OnesInARow  other are very heavy, this one will allow all numbers less than 0x03FFFFFFFFFFFFFF
        "--" ARG_EFILTER"=64",  //SmartFilterFlags::BytesRepeat4

        "--" ARG_FMATCH"=0",
    };

    auto args = console::parse_command_line(sizeof(commandline)/ sizeof(char*), commandline); //console::parse_command_line(argc, argv);
    if (!args.isValid())
    {
        return 1;
    }

    if (!CUDA_check_keeloq_works())
    {
        printf("Error: This device cannot compute keeloq right. Single ecnryption and decryption mismacth.");
        assert(false);
        return 1;
    }


    // Could be used for device check and input validation
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    bruteforce(args);

    // this will free all memory aswell
    cudaDeviceReset();
}
