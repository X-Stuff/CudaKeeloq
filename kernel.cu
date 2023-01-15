#include <vector>
#include <chrono>
#include <string>
#include <conio.h>
#include <stdio.h>
#include <stdlib.h>

#include "kernel.cuh"
#include "keeloq_main.cuh"
#include "keeloq_generators.cuh"

#include "CUDA_helpers.cuh"

#define console_clear() printf("\033[H\033[J")

#define console_cursor_up(lines) printf("\033[%dA", (lines))

#define console_cursor_ret_up() printf("\033[F")
#define console_set_cursor(x,y) printf("\033[%d;%dH", (y), (x))

#define save_cursor_pos() printf("\033[s")
#define load_cursor_pos() printf("\033[u")


constexpr int NUM_BLOCKS = 32;
constexpr int NUM_THREAD = 256;

// How much interation per one thread (basically for loop count)
constexpr int NUM_DECRYPTORS_PER_THREAD = 128;


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

void print_decrypted_array(const SingleResult::DecryptedArray& array, KeeloqLearningType learning_match = KeeloqLearningType::INVALID)
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

    SingleResult::DecryptedArray all_dec = keeloq_decrypt(ota, man);
    print_decrypted_array(all_dec);
}


KernelResult run_block(KernelInput& input)
{
    int error = CUDA_generator_wrapper<NUM_BLOCKS, NUM_THREAD>(input);
    assert(error == 0);

    return CUDA_keeloq_main_wrapper<NUM_BLOCKS, NUM_THREAD>(input);
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

void progress_bar(double percent, const std::chrono::seconds& elapsed)
{
    constexpr auto progress_width = 80;
    static char progress_fill[progress_width] = {0};
    static char progress_none[progress_width] = {0};
    if (progress_fill[0] == 0)
    {
        memset(progress_fill, '=', sizeof(progress_fill));
        memset(progress_none, '-', sizeof(progress_none));
    }

    printf("[%.*s>", (int)(progress_width * percent), progress_fill);
    printf("%.*s]", (int)(progress_width * (1 - percent)), progress_none);
    printf("%d%%  %02lld:%02lld:%02lld   \n", (int)(percent * 100),
        elapsed.count() / 3600, (elapsed.count() / 60) % 60, elapsed.count() % 60);
}

int main(int argc, char** argv)
{
    //test_keeloq(); return;

    CUDA_test_generator_alphabet();
    return;

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
        BruteforceConfig dict_config(BruteforceConfig::Type::Dictionary, dictionary.size());
        BruteforceConfig brute_config(0xCEB6AE4800000000,  BruteforceConfig::Type::Simple, 0xFFFFFFFF);

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
                kernel_input.WriteDecryptors(dictionary, batch * key_per_batch, key_per_batch);
            }
            else {
                kernel_input.UpdateInitialDecryptor();
            }

            auto kernel_result = run_block(kernel_input);
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

            progress_bar(progress_percent, overall);
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
