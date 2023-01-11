#include <vector>

#include "stdio.h"
#include "stdlib.h"

#include "keeloq.cuh"
#include "keeloq_generators.cuh"

#include "CUDA_helpers.cuh"
#include "CUDA_check.cuh"

constexpr int NUM_BLOCKS = 32;
constexpr int NUM_THREAD = 256;

// How much interation per one thread (basically for loop count)
constexpr int NUM_DECRYPTORS_PER_THREAD = 512;


struct CudaRunSetup
{
    // Bruteforce setup
    CudaRunSetup(std::vector<EncData>&& data, const DectyptorGenerationConfig& gen, uint32_t blocks, uint32_t threads, uint32_t iterations)
        : encrypted_data(std::move(data)), config(gen)
    {
        uint32_t num_decryptors_per_run = iterations * threads * blocks;

        // allocated once. updated every run on GPU
        decryptors = std::vector<Decryptor>(num_decryptors_per_run);

        // allocated once. updated evert run on GPU. copied to CPU only if match found.
        block_results = std::vector<SingleResult>(encrypted_data.size() * decryptors.size());

        alloc();
    }

    // Dictionary setup
    CudaRunSetup(std::vector<EncData>&& data, uint32_t blocks, uint32_t threads, uint32_t iterations)
        : CudaRunSetup(std::move(data), DectyptorGenerationConfig(0, GeneratorType::Dictionary), blocks, threads, iterations)
    {
    }

    ~CudaRunSetup()
    {
        free();
    }

    KernelInput CreateInputs(size_t batch_num, const std::vector<Decryptor>* dictionary = nullptr)
    {
        if (dictionary != nullptr)
        {
            size_t start_decryptor = batch_num * DecryptorsPerBatch();
            size_t batch_decryptors_num = max(0ull, min(DecryptorsPerBatch(), dictionary->size() - start_decryptor));

            CUDA_decryptors->write(&(*dictionary)[start_decryptor], batch_decryptors_num);
        }

        return KernelInput(CUDA_encrypted, CUDA_decryptors, CUDA_results, config);
    }

    const std::vector<SingleResult>& ReadResults()
    {
        CUDA_results->copy(block_results);
        return block_results;
    }

    inline size_t DecryptorsPerBatch() const {
        return decryptors.size();
    }

private:

    void alloc()
    {
        //
        assert(CUDA_encrypted == nullptr && "Encrypted data already allocated on GPU");
        assert(CUDA_decryptors == nullptr && "Decryptors data already allocated on GPU");
        assert(CUDA_results == nullptr && "Results data already allocated on GPU");

        // ALLOCATE ON GPU
        if (CUDA_encrypted == nullptr)
        {
            CUDA_encrypted  = CUDA_Array<EncData>::allocate(encrypted_data);
        }

        if (CUDA_decryptors == nullptr)
        {
            CUDA_decryptors = CUDA_Array<Decryptor>::allocate(decryptors);
        }

        if (CUDA_results == nullptr)
        {
            CUDA_results    = CUDA_Array<SingleResult>::allocate(block_results);
        }
    }

    void free()
    {
        if (CUDA_encrypted != nullptr)
        {
            CUDA_encrypted->free();
            CUDA_encrypted = nullptr;
        }

        if (CUDA_decryptors == nullptr)
        {
            CUDA_decryptors->free();
            CUDA_decryptors = nullptr;
        }

        if (CUDA_results == nullptr)
        {
            CUDA_results->free();
            CUDA_results = nullptr;
        }

        encrypted_data.clear();
        decryptors.clear();
        block_results.clear();
    }

private:

    //
    DectyptorGenerationConfig config;

    // Constant per run
    std::vector<EncData> encrypted_data;
    CUDA_Array<EncData>* CUDA_encrypted = nullptr;

    // could be pretty much data here
    std::vector<Decryptor> decryptors;
    CUDA_Array<Decryptor>* CUDA_decryptors = nullptr;

    // could be pretty much data here
    std::vector<SingleResult> block_results;
    CUDA_Array<SingleResult>* CUDA_results = nullptr;
};


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

    printf(".");
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
        CudaRunSetup setup(std::move(otas), NUM_BLOCKS, NUM_THREAD, NUM_DECRYPTORS_PER_THREAD);

        size_t dict_size = dictionary.size();
        size_t block_runs = dict_size / setup.DecryptorsPerBatch() + 1;

        for (size_t block = 0; block < block_runs; ++block)
        {
            KernelInput MainKernelInputs = setup.CreateInputs(block, &dictionary);

            auto kernel_result = run_block(MainKernelInputs);
            if (process_block_results(kernel_result, setup))
            {
                break;
            }
        }

    }
    cudaDeviceReset();
}
