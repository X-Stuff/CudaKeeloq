
#include <vector>

#include "stdio.h"
#include "stdlib.h"

#include "keeloq.cuh"
#include "keeloq_generators.cuh"

#include "CUDA_helpers.cuh"
#include "CUDA_check.cuh"

#define STRICT_ANALYSIS 1


void print_decrypted_array(const DecryptedArray& array, uint8_t learning_match = KEELOQ_LEARNING_INVALID)
{
    for (uint8_t i = 0; i < KEELOQ_LEARNING_LAST; ++i)
    {
        const char* name = LearningNames[i];

        uint32_t dec = array.data[i];
        if (dec != 0)
        {
            uint32_t btn = dec >> 28;
            uint32_t srl = (dec >> 16) & 0x3ff;
            uint32_t cnt = dec & 0xFFFF;

            printf("[%-40s] Btn:0x%X\tSerial:0x%X\tCounter:0x%X\t%s\n", name, btn, srl, cnt,
                (i == learning_match ? "(MATCH)" : ""));
        }
        else
        {
            printf("[%-40s] SKIPPED\n", name);
        }
    }
}

void print_result(const SingleResult& result)
{
    printf("Results:\n\tOTA: 0x%llX\tMan key: 0x%llX\n\n", result.ota, result.man);
    print_decrypted_array(result.results, result.match);
}

void test_keeloq()
{
    // Using encrypt and decrypt functions:
    uint64_t man = 0xCEB6AE48B5C63ED2; // benica manuf
    uint64_t ota = 0xCCA9B335A81FD504; //

    DecryptedArray all_dec = keeloq_decrypt(ota, man);
    print_decrypted_array(all_dec);
}

int main(int argc, char** argv)
{
    constexpr int NUM_BLOCKS = 16;
    constexpr int NUM_THREAD = 256;


    //test_keeloq(); return;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    std::vector<uint64_t> otas = {
        0xC65D52A0A81FD504,
        0xCCA9B335A81FD504,
    };

    std::vector<Decryptor> mans = {
        { 0xCEB6AE48B5C63ED1, 0 },
        { 0xCEB6AE48B5C63ED2, 0 },
        { 0xCEB6AE48B5C63ED3, 0 },
    };

    std::vector<SingleResult> results(otas.size() * mans.size());
    std::vector<MatchData> matches(results.size());

    KernelInput MainKernelInputs = KernelInput();

    MainKernelInputs.encdata    = CUDA_Array<EncData>::allocate(otas);
    MainKernelInputs.decryptors = CUDA_Array<Decryptor>::allocate(mans);
    MainKernelInputs.results    = CUDA_Array<SingleResult>::allocate(results);
    MainKernelInputs.matches    = CUDA_Array<MatchData>:: allocate(matches);

    MainKernelInputs.generation.type = GeneratorType::None;

    int num_matches = 0;
    int num_errors = 0;

    int error = CUDA_generator_wrapper<NUM_BLOCKS, NUM_THREAD>(MainKernelInputs);
    assert(error == 0);

    CUDA_keeloq_main_wrapper<NUM_BLOCKS, NUM_THREAD>(MainKernelInputs, num_matches, num_errors);
    if (num_matches > 0)
    {
        MainKernelInputs.results->copy(results);
        MainKernelInputs.matches->copy(matches);

        for (const auto& match : matches)
        {
            if (match >= 0)
            {
                if (match < results.size())
                {
                    printf("------------------\n");
                    print_result(results[match]);
                }
                else
                {
                    printf(
                        "INVALID MATCH INDEX: %d. results. size:%zd.\n"
                        "Generator data:\n"
                        "\tType:%d\tMAN:0x%llX\tSEED:0x%X\n",
                            match, results.size(),
                            MainKernelInputs.generation.type,
                            MainKernelInputs.generation.initial.man,
                            MainKernelInputs.generation.initial.seed);
                }
            }
        }
    }
    else if (num_errors == 0)
    {
        printf("NO MATCHES\n");
    }

    if (num_errors != 0)
    {
        printf("Kernel returned error: %d\n", num_errors);
    }

    cudaDeviceReset();
    getchar();
}
