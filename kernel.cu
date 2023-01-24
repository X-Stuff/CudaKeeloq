#define CU_FILE

#include <vector>
#include <chrono>
#include <string>
#include <stdlib.h>

#include "host/console.h"

#include "kernel.cuh"
#include "keeloq_main.cuh"
#include "keeloq_generators.cuh"

#include "CUDA_helpers.cuh"


std::string CudaRunSetup::GetLearningTypeName() const
{
    if (kernel_inputs.learning_types[KeeloqLearningType::LAST])
    {
        return "ALL";
    }

    char str[512];
    int len = 0;
    for (auto type = 0; type < KeeloqLearningType::LAST; ++type)
    {
        if (kernel_inputs.learning_types[type])
        {
            len += sprintf_s(&str[len], sizeof(str) - len, "%s, ", KeeloqLearningType::Name(type));
        }
    }

    return std::string(str, len - 2);
}

std::string CudaRunSetup::ToString() const
{
    assert(inited);

    char tmp[1024];
    sprintf_s(tmp, "Setup:\n"
        "\tCUDA: Blocks:%u Threads:%u Iteraions:%u\n"
        "\tEncrypted data size:%zd\n"
        "\tLearning type:%s\n"
        "\tResults per batch:%zd\n"
        "\tDecryptors per batch:%zd\n"
        "\tConfig: %s",
        CudaBlocks(), CudaThreads(), CudaThreadIterations(),
        encrypted_data.size(), GetLearningTypeName().c_str(), ResultsPerBatch(), KeysCheckedInBatch(), Config().toString().c_str());

    return std::string(tmp);
}


void CudaRunSetup::alloc()
{
    //
    assert(kernel_inputs.encdata    == nullptr && "Encrypted data already allocated on GPU");
    assert(kernel_inputs.decryptors == nullptr && "Decryptors data already allocated on GPU");
    assert(kernel_inputs.results    == nullptr && "Results data already allocated on GPU");

    // ALLOCATE ON GPU
    if (kernel_inputs.encdata == nullptr)
    {
        kernel_inputs.encdata  = CUDA_Array<EncData>::allocate(encrypted_data);
    }

    if (kernel_inputs.decryptors == nullptr)
    {
        kernel_inputs.decryptors = CUDA_Array<Decryptor>::allocate(decryptors);
    }

    if (kernel_inputs.results == nullptr)
    {
        kernel_inputs.results    = CUDA_Array<SingleResult>::allocate(block_results);
    }
}

void CudaRunSetup::free()
{
    if (kernel_inputs.encdata != nullptr)
    {
        kernel_inputs.encdata->free();
        kernel_inputs.encdata = nullptr;
    }

    if (kernel_inputs.decryptors != nullptr)
    {
        kernel_inputs.decryptors->free();
        kernel_inputs.decryptors = nullptr;
    }

    if (kernel_inputs.results != nullptr)
    {
        kernel_inputs.results->free();
        kernel_inputs.results = nullptr;
    }

    encrypted_data.clear();
    decryptors.clear();
    block_results.clear();
}

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
    if (args.selected_learning.size() == 0)
    {
        printf("Bruteforcing without specific learning type (slower)"
            "(1 KKey/s == %u Kkc (keeloq calcs) per second)\n", KeeloqLearningType::LAST);
    }

    for (const auto& config: args.brute_configs)
    {
        CudaRunSetup single_run(args.inputs, config, args.selected_learning, args.cuda_blocks, args.cuda_threads, args.cuda_loops);

        printf("\nallocating...");
        single_run.Init();

        printf("\rRunning...\t\t\t\n%s\n", single_run.ToString().c_str());

        bool match = false;

        size_t num_batches = single_run.NumBatches();
        size_t key_per_batch = single_run.KeysCheckedInBatch();

        auto dec_start = std::chrono::system_clock::now();

        for (size_t batch = 0; !match && batch < num_batches; ++batch)
        {
            auto batch_start = std::chrono::high_resolution_clock::now();

            KernelInput& kernel_input = single_run.Inputs();

            if (single_run.Type() == BruteforceType::Dictionary) {
                kernel_input.WriteDecryptors(config.decryptors, batch * key_per_batch, key_per_batch);
            }
            else {
                kernel_input.UpdateInitialDecryptor();
            }

            int error = CUDA_generator_wrapper(kernel_input, single_run.CudaBlocks(), single_run.CudaThreads());
            assert(error == 0);
            auto kernel_result = CUDA_keeloq_main_wrapper(kernel_input, single_run.CudaBlocks(), single_run.CudaThreads());
            match = process_block_results(kernel_result, single_run);

            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - batch_start);

            if (batch == 0 || match)
            {
                console_hide_cursor();
                printf("\n\n\n");
            }

            if (!match)
            {
                auto kilo_result_per_second = duration.count() == 0 ? 0 : key_per_batch / duration.count();
                auto progress_percent = (double)(batch + 1)/ num_batches;

                console_cursor_ret_up(2);

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
    const char* commandline[] = {
        "tests",
        "--" ARG_INPUTS"=0xC65D52A0A81FD504,0xCCA9B335A81FD504,0xE0DA7372A81FD504",
        "--" ARG_MODE"=2,0,1,3",
        //"--" ARG_LTYPE"=6",

        "--" ARG_WORDDICT"=0xCEB6AE48B5C63ED1,0xCEB6AE48B5C63ED2,0xCEB6AE48B5C63ED3",

#if _DEBUG
        "--" ARG_BLOCKS"=512",
        "--" ARG_LOOPS"=2",
        "--" ARG_START"=0xCEB6AE48B0000000",
#else
        "--" ARG_BLOCKS"=1024",
        "--" ARG_LOOPS"=2",
        "--" ARG_START"=0xCEB6AE4800000000",
#endif
        "--" ARG_COUNT"=0xFFFFFFFF",

        // "--" ARG_IFILTER"=0x2", include filter let be all (otherwise will have big impact)
        "--" ARG_EFILTER"=96",  // BytesRepeat4 | BytesIncremental should increse performance(?)

        "--" ARG_ALPHABET"=examples/alphabet.bin,CE:B6:AE:48:B5:C6:3E:D2",//:AA:BB:CC:DD:EE:FF:00:11",

        "--" ARG_FMATCH"=0",

        "--" ARG_TEST"=1",
    };

    console_set_width(CONSOLE_WIDTH);

    auto args = console::parse_command_line(sizeof(commandline)/ sizeof(char*), commandline); //console::parse_command_line(argc, argv);
    if (args.run_tests)
    {
        printf("\n...RUNNING TESTS...\n");
        console::tests::run();
        generators::tests::run();
        printf("\n...TESTS FINISHED...\n");

        console_clear();
        //return;
    }

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

    // Just for test
    if (args.brute_configs[0].type == BruteforceType::Alphabet)
    {
        args.brute_configs[0].start.man = 0;
        args.brute_configs[0].next.man = 0;
    }

    bruteforce(args);

    // this will free all memory aswell
    cudaDeviceReset();
}
