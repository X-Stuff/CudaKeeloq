
#include <vector>

#include "stdio.h"
#include "stdlib.h"

#include "keeloq.cuh"
#include "CUDA_Vec.cuh"
#include "CUDA_check.cuh"

#define STRICT_ANALYSIS 1


struct SingleResult
{
    DecryptedArray results;

    uint64_t man;
    uint32_t seed;

    uint64_t ota;

    int8_t match;
};

struct CUDACtx
{
    uint32_t thread_max;
    uint32_t thread_id;
};

typedef uint64_t EncData;
typedef uint32_t MatchData; // now just index in results vector

struct Decryptor
{
    uint64_t man;
    uint32_t seed;
};

struct CUDA_MAIN_Inputs
{
    CUDA_VEC<EncData>* encdata;
    CUDA_VEC<Decryptor>* decryptors;

    CUDA_VEC<SingleResult>* results;
    CUDA_VEC<MatchData>* matches;

    void free()
    {
        if (encdata)
        {
            encdata->free();
            encdata = nullptr;
        }

        if (decryptors)
        {
            decryptors->free();
            decryptors = nullptr;
        }

        if (matches)
        {
            matches->free();
            matches = nullptr;
        }

        if (results)
        {
            results->free();
            results = nullptr;
        }
    }
};


__device__ __host__ void run_keeloq_decryption(const CUDACtx& ctx, CUDA_VEC<EncData>* encrypted, CUDA_VEC<Decryptor>* decryptors, CUDA_VEC<SingleResult>* results) {

    assert(encrypted);
    assert(decryptors);
    assert(results);

    for (uint32_t decryptor_index = ctx.thread_id; decryptor_index < decryptors->num; decryptor_index += ctx.thread_max)
    {
        uint64_t man  = decryptors->CUDA_data[decryptor_index].man;
        uint32_t seed = decryptors->CUDA_data[decryptor_index].seed;

        for (uint32_t ota_index = 0; ota_index < encrypted->num; ota_index++)
        {
            size_t result_index = decryptor_index * encrypted->num + ota_index;

            SingleResult& result = results->CUDA_data[result_index];

            result.man = man;
            result.seed = seed;
            result.match = -1;

            result.ota = encrypted->CUDA_data[ota_index];

            result.results = keeloq_decrypt(result.ota, result.man, result.seed);
        }
    }
}

__device__ uint8_t analyze_results_cnt(SingleResult* results, uint32_t num, uint8_t learning_type)
{
    uint8_t bit_tolerance = 7; //(1 << ceilf(__log2f(num))) - 1; // if num == 5, ceil(log2(5)) == 3, (1 << 3) - 1 == 7, 7 == 0b0111

    uint32_t expected_btn = results[0].results.data[learning_type] >> 28;
    uint32_t lrn_matches = 0;

    for (uint8_t item = 0; item < num; ++item)
    {
        uint32_t btn = results[item].results.data[learning_type] >> 28;
        lrn_matches += (btn ^ expected_btn) <= bit_tolerance;
    }

    return lrn_matches == num;
}

__device__ uint8_t analyze_results_btn(SingleResult* results, uint32_t num, uint8_t learning_type, uint8_t bit_tolerance = 0)
{
    uint32_t expected_btn = results[0].results.data[learning_type] >> 28;
    uint32_t lrn_matches = 1;

    for (uint8_t item = 1; item < num; ++item)
    {
        uint32_t btn = results[item].results.data[learning_type] >> 28;
        lrn_matches += (btn ^ expected_btn) <= bit_tolerance;
    }

    return lrn_matches == num;
}

__device__ uint8_t analyze_results_srl(SingleResult* results, uint32_t num)
{
    uint8_t match_count = 0; // 0 or 1. if bigger - double match
    uint8_t match_learning_type = KEELOQ_LEARNING_INVALID;

    SingleResult& first = results[0];

    for (uint8_t lrn = 0; lrn < KEELOQ_LEARNING_LAST; ++lrn)
    {
        uint32_t expected_srl = (first.results.data[lrn] >> 16) & 0x3ff;
        uint32_t lrn_matches = 1;

        for (uint8_t i = 1; i < num; ++i)
        {
            SingleResult& item = results[i];

            uint32_t srl = (item.results.data[lrn] >> 16) & 0x3ff;
            lrn_matches += srl == expected_srl && srl != 0;
        }

        // ifs are bad
        bool has_match = lrn_matches == num;

        match_count += has_match;
        match_learning_type = has_match * lrn + !has_match * match_learning_type;

#if !STRICT_ANALYSIS
        if (match_count)
        {
            break;
        }
#endif
    }

    return match_learning_type;
}

__device__ uint8_t find_matches(const CUDACtx& ctx, CUDA_VEC<SingleResult>* results, uint32_t num_decryptors, uint32_t num_inputs)
{
    uint8_t result_error = 0;

    for (uint32_t decryptor_index = ctx.thread_id; decryptor_index < num_decryptors; decryptor_index += ctx.thread_max)
    {
        SingleResult* result_start = &results->CUDA_data[decryptor_index * num_inputs];

        // ifs are bad
        uint8_t learning_type = analyze_results_srl(result_start, num_inputs);
        if (learning_type != KEELOQ_LEARNING_INVALID)
        {
            if (analyze_results_btn(result_start, num_inputs, learning_type) &&  // same button
                analyze_results_cnt(result_start, num_inputs, learning_type))
            {
                for (int i = 0; i < num_inputs; ++i)
                {
                    result_start[i].match = learning_type;
                }
            }
        }

#if STRICT_ANALYSIS
        uint64_t instance_man = results->CUDA_data[decryptor_index * num_inputs].man;
        for (int i = 0; i < num_inputs; ++i)
        {
            result_error += instance_man != result_start[i].man;
        }
#endif
    }

    return result_error;
}

__device__ uint8_t analyze_results(const CUDACtx& ctx, const CUDA_VEC<SingleResult>* results, CUDA_VEC<MatchData>* matches)
{
    uint8_t num_matches = 0;

    for (uint32_t r = ctx.thread_id; r < results->num; r += ctx.thread_max)
    {
        const SingleResult& result = results->CUDA_data[r];

        // Writing index in results if match is valid
        uint8_t is_match = (result.match != KEELOQ_LEARNING_INVALID);
        matches->CUDA_data[r] = r * is_match;

        num_matches += is_match;
    }

    return num_matches;
}

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


__global__ void CUDA_main(CUDA_MAIN_Inputs* CUDA_inputs, int* ret)
{
    CUDACtx ctx = {
        gridDim.x * blockDim.x,                 // thread_max
        blockIdx.x * blockDim.x + threadIdx.x   // thread_id
    };

    if (ctx.thread_id == 0) {

        uint32_t pln_test = 0x11223344;
        uint64_t key_test = 0xDEADBEEF00226688;
        uint32_t enc_test = keeloq_common_encrypt(pln_test, key_test);
        if (pln_test != keeloq_common_decrypt(enc_test, key_test))
        {
            ret[ctx.thread_id] = -42;
            return;
        }
    }

    run_keeloq_decryption(ctx, CUDA_inputs->encdata, CUDA_inputs->decryptors, CUDA_inputs->results);

    uint8_t num_errors = find_matches(ctx, CUDA_inputs->results, CUDA_inputs->decryptors->num, CUDA_inputs->encdata->num);


    ret[ctx.thread_id] += analyze_results(ctx, CUDA_inputs->results, CUDA_inputs->matches);
}


template<uint16_t ThreadBlocks, uint16_t ThreadsInBlock>
int CUDA_main_wrapper(CUDA_MAIN_Inputs& mainInputs)
{
    DOUBLE_ARRAY<int> kernel_result(ThreadBlocks * ThreadsInBlock);

    CUDA_main<<<ThreadBlocks, ThreadsInBlock>>>(&mainInputs, kernel_result.CUDA_mem);

    kernel_result.read_GPU();

    return kernel_result.HOST_mem[0];
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

    std::vector<Decryptor> mans = {
        { 0xCEB6AE48B5C63ED1, 0 },
        { 0xCEB6AE48B5C63ED2, 0 },
        { 0xCEB6AE48B5C63ED3, 0 },
    };

    std::vector<SingleResult> results(otas.size() * mans.size());
    std::vector<MatchData> matches(results.size());

    CUDA_MAIN_Inputs MainKernelInputs = {0};
    MainKernelInputs.encdata    = CUDA_VEC<EncData>::allocate(otas);
    MainKernelInputs.decryptors = CUDA_VEC<Decryptor>::allocate(mans);
    MainKernelInputs.results    = CUDA_VEC<SingleResult>::allocate(results);
    MainKernelInputs.matches    = CUDA_VEC<MatchData>:: allocate(matches);


    int error = CUDA_main_wrapper<32,256>(MainKernelInputs);
    if (error == 0)
    {
        MainKernelInputs.results->copy(results);
        MainKernelInputs.matches->copy(matches);

        for (const auto& match : matches)
        {
            printf("------------------\n");
            print_result(results[match]);
        }
    }
    else
    {
        printf("Kernel returned error: %d\n", error);
    }

    cudaDeviceReset();
    getchar();
}
