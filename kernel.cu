#include <vector>
#include <chrono>
#include <string>

#include "stdio.h"
#include "stdlib.h"

#include "kernel.cuh"
#include "keeloq_main.cuh"
#include "keeloq_generators.cuh"

#include "CUDA_helpers.cuh"

#define console_clear() printf("\033[H\033[J")
#define console_set_cursor(x,y) printf("\033[%d;%dH", (y), (x))


constexpr int NUM_BLOCKS = 16;
constexpr int NUM_THREAD = 256;

// How much interation per one thread (basically for loop count)
constexpr int NUM_DECRYPTORS_PER_THREAD = 32;


void print_decrypted_data(uint32_t data, const char* name, bool ismatch)
{
    if (data != 0)
    {
        uint32_t btn = data >> 28;
        uint32_t srl = (data >> 16) & 0x3ff;
        uint32_t cnt = data & 0xFFFF;

        printf("[%-40s] Btn:0x%X\tSerial:0x%X\tCounter:0x%X\t%s\n", name, btn, srl, cnt,
            (ismatch ? "(MATCH)" : ""));
    }
    else
    {
        printf("[%-40s] SKIPPED\n", name);
    }
}

void print_decrypted_array(const DecryptedArray& array, KeeloqLearningType learning_match = KeeloqLearningType::INVALID)
{
    for (uint8_t i = 0; i < KeeloqLearningType::LAST; ++i)
    {
        const char* name = LearningNames[i];
        print_decrypted_data(array.data[i], name, learning_match);
    }
}

void print_result(const SingleResult& result, bool onlymatch = true)
{
    printf("Results:\n\tOTA: 0x%llX\tMan key: 0x%llX\n\n", result.ota, result.man);

    bool print_all = result.match == KeeloqLearningType::INVALID || !onlymatch;
    if (print_all)
    {
        print_decrypted_array(result.results, result.match);
    }
    else
    {
        const char* name = LearningNames[result.match % KeeloqLearningType::LAST];
        print_decrypted_data(result.results.data[result.match % KeeloqLearningType::LAST], name, true);
    }

}

void test_keeloq()
{
    // Using encrypt and decrypt functions:
    uint64_t man = 0xCEB6AE48B5C63ED2; // benica manuf
    uint64_t ota = 0xCCA9B335A81FD504; //

    DecryptedArray all_dec = keeloq_decrypt(ota, man);
    print_decrypted_array(all_dec);
}


KernelResult run_block(KernelInput& input)
{
    int num_matches = 0;
    int num_errors = 0;

    int error = CUDA_generator_wrapper<NUM_BLOCKS, NUM_THREAD>(input);
    assert(error == 0);

    return CUDA_keeloq_main_wrapper<NUM_BLOCKS, NUM_THREAD>(input, num_matches, num_errors);
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
            print_result(result);
        }

        return true;
    }

    return false;
}

int main(int argc, char** argv)
{
    //test_keeloq(); return;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    std::vector<uint64_t> otas = {
        0xC65D52A0A81FD504,
        0xCCA9B335A81FD504,
    };

    std::vector<Decryptor> dictionary = {
        { 0xCEB6AE48B5C63ED1, 0 },
        { 0xCEB6AE48B5C63ED2, 0 },
        { 0xCEB6AE48B5C63ED3, 0 },
    };

    assert(CUDA_check_keeloq_works());
    {
        DectyptorGenerationConfig dict_config(GeneratorType::Dictionary, dictionary.size());
        DectyptorGenerationConfig brute_config(0xCEB6AE48B5000000,  GeneratorType::Brute, 0x00FFFFFF);

        // keep them inside this {} scope, otherwise free() will cause errors because of cudaDeviceReset()
        CudaRunSetup dict_setup(otas, dict_config,   NUM_BLOCKS, NUM_THREAD, NUM_DECRYPTORS_PER_THREAD);
        CudaRunSetup brute_setup(otas, brute_config, NUM_BLOCKS, NUM_THREAD, NUM_DECRYPTORS_PER_THREAD);

        CudaRunSetup& setup = brute_setup;// dict_setup; //
        setup.Init();

        printf("%s\n(1 KKey/s == %u Kkc (keeloq calcs) per second)\n",
            setup.ToString().c_str(), KeeloqLearningType::LAST);

        bool match = false;

        size_t num_batches = setup.NumBatches();
        size_t key_per_batch = setup.KeysCheckedInBatch();

        for (size_t batch = 0; !match && batch < num_batches; ++batch)
        {
            auto start = std::chrono::high_resolution_clock::now();

            KernelInput& kernel_input = setup.Inputs();

            if (setup.Type() == GeneratorType::Dictionary) {
                kernel_input.WriteDecryptors(dictionary, batch * key_per_batch, key_per_batch);
            }
            else {
                kernel_input.UpdateInitialDecryptor();
            }

            auto kernel_result = run_block(kernel_input);
            match = process_block_results(kernel_result, setup);

            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - start);

            auto kilo_result_per_second = duration.count() == 0 ? 0 : key_per_batch / duration.count();

            printf("\r[%c]\t Elapsed: %llu(ms) Completed: %d%% Speed: %llu KKeys/s", WAIT_CHAR(batch),
                duration.count(), (int)(((double)(batch + 1)/ num_batches) * 100),
                kilo_result_per_second);
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
