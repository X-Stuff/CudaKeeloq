#include <vector>
#include <chrono>
#include <string>
#include <stdlib.h>

#include "console.cuh"
#include "kernel.cuh"
#include "keeloq_main.cuh"
#include "keeloq_generators.cuh"

#include "CUDA_helpers.cuh"

constexpr int NUM_BLOCKS = 32;
constexpr int NUM_THREAD = 256;

// How much interation per one thread (basically for loop count)
constexpr int NUM_DECRYPTORS_PER_THREAD = 128;

void test_keeloq()
{
    // Using encrypt and decrypt functions:
    uint64_t man = 0xCEB6AE48B5C63ED2; // benica manuf
    uint64_t ota = 0xCCA9B335A81FD504; //

    SingleResult::DecryptedArray all_dec = keeloq_decrypt(ota, man);
    all_dec.print();
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

            printf("------------------\n");
            result.print();
        }

        return true;
    }

    return false;
}

int main(int argc, const char** argv)
{
    //test_keeloq(); return;
    auto targs = console::tests::run();
    return;

    auto args = console::parse_command_line(argc, argv);
    if (!args.isValid())
    {
        return 1;
    }


    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    std::vector<uint64_t> otas = {
        0xC65D52A0A81FD504,
        0xCCA9B335A81FD504,
        0xE0DA7372A81FD504
    };

    std::vector<Decryptor> dictionary = {
        { 0xCEB6AE48B5C63ED1, 0 },
        { 0xCEB6AE48B5C63ED2, 0 },
        { 0xCEB6AE48B5C63ED3, 0 },
    };

    assert(CUDA_check_keeloq_works());
    {
        auto dict_config = BruteforceConfig::GetDictionary(std::move(dictionary));
        auto brute_config = BruteforceConfig::GetBruteforce(0xCEB6AE4800000000, 0xFFFFFFFF);

        // keep them inside this {} scope, otherwise free() will cause errors because of cudaDeviceReset()
        CudaRunSetup dict_setup(otas, dict_config,   NUM_BLOCKS, NUM_THREAD, NUM_DECRYPTORS_PER_THREAD);
        CudaRunSetup brute_setup(otas, brute_config, NUM_BLOCKS, NUM_THREAD, NUM_DECRYPTORS_PER_THREAD);

        CudaRunSetup& setup = brute_setup;//dict_setup; //
        setup.Init();

        printf("%s\n(1 KKey/s == %u Kkc (keeloq calcs) per second)\n",
            setup.ToString().c_str(), KeeloqLearningType::LAST);

        bool match = false;

        size_t num_batches = setup.NumBatches();
        size_t key_per_batch = setup.KeysCheckedInBatch();

        auto dec_start = std::chrono::system_clock::now();

        for (size_t batch = 0; !match && batch < num_batches; ++batch)
        {
            auto batch_start = std::chrono::high_resolution_clock::now();

            KernelInput& kernel_input = setup.Inputs();

            if (setup.Type() == BruteforceConfig::Type::Dictionary) {
                kernel_input.WriteDecryptors(dict_config.decryptors, batch * key_per_batch, key_per_batch);
            }
            else {
                kernel_input.UpdateInitialDecryptor();
            }

            int error = CUDA_generator_wrapper<NUM_BLOCKS, NUM_THREAD>(kernel_input);
            assert(error == 0);
            auto kernel_result = CUDA_keeloq_main_wrapper<NUM_BLOCKS, NUM_THREAD>(kernel_input);
            match = process_block_results(kernel_result, setup);

            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - batch_start);

            if (batch == 0 || match)
            {
                printf("\n");
                save_cursor_pos();
            }

            auto kilo_result_per_second = duration.count() == 0 ? 0 : key_per_batch / duration.count();
            auto progress_percent = (double)(batch + 1)/ num_batches;

            load_cursor_pos();
            printf("[%c][%zd/%zd]\t %llu(ms)/batch Speed: %llu KKeys/s\n", WAIT_CHAR(batch),
                batch, num_batches, duration.count(),
                kilo_result_per_second);

            auto overall = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::system_clock::now() - dec_start);

            console::progress_bar(progress_percent, overall);
        }

        if (!match)
        {
            printf("\n\nAfter: %zd batches no results was found. Keys checked:%zd\n\n",
                num_batches, num_batches * key_per_batch);
        }

    }

    // this will free all memory aswell
    cudaDeviceReset();
}
