#pragma once

#include <vector>
#include <assert.h>

#include "keeloq_main.cuh"



constexpr char WAIT_SPIN[] = "|/-\\";

#define WAIT_CHAR(i) (WAIT_SPIN[i % (sizeof(WAIT_SPIN) - 1)])

struct CudaRunSetup
{
    CudaRunSetup(const std::vector<EncData>& data, const BruteforceConfig& gen, uint32_t blocks, uint32_t threads, uint32_t iterations)
        : encrypted_data(data)
    {
        CUDASetup[0] = blocks;
        CUDASetup[1] = threads;
        CUDASetup[2] = iterations;

        num_decryptors_per_batch = iterations * threads * blocks;

        kernel_inputs.generator = gen;
    }

    ~CudaRunSetup()
    {
        free();
    }

    const std::vector<SingleResult>& ReadResults()
    {
        kernel_inputs.results->copy(block_results);
        return block_results;
    }

    const std::vector<Decryptor>& ReadDecryptors()
    {
        kernel_inputs.decryptors->copy(decryptors);
        return decryptors;
    }

    // Allocates memory
    inline void Init()
    {
        if (!inited)
        {
            // allocated once. updated every run on GPU
            decryptors = std::vector<Decryptor>(num_decryptors_per_batch);

            // allocated once. updated evert run on GPU. copied to CPU only if match found.
            block_results = std::vector<SingleResult>(encrypted_data.size() * decryptors.size());

            alloc();

            inited = true;
        }
    }

    inline uint32_t CudaBlocks() const { return CUDASetup[0]; }

    inline uint32_t CudaThreads() const { return CUDASetup[1]; }

    inline uint32_t CudaThreadIterations() const { return CUDASetup[2]; }

    inline const BruteforceConfig& Config() const { assert(inited); return kernel_inputs.generator; }

    inline BruteforceConfig::Type Type() { assert(inited); return Config().type; }

    inline KernelInput& Inputs() { assert(inited); return kernel_inputs; }

    inline size_t NumBatches() {
        assert(inited);
        if (Type() == BruteforceConfig::Type::Dictionary) {
            uint8_t non_align = Config().dict_size() % KeysCheckedInBatch() == 0 ? 0 : 1;
            return Config().dict_size() / KeysCheckedInBatch() + non_align;
        }
        else {
            uint8_t non_align = Config().dict_size() % KeysCheckedInBatch() == 0 ? 0 : 1;
            return Config().brute_size() / KeysCheckedInBatch() + non_align;
        }
    }

    inline size_t KeysCheckedInBatch() const {
        assert(inited);
        return decryptors.size();
    }

    inline size_t ResultsPerBatch() const {
        assert(inited);
        return block_results.size();
    }

    std::string ToString()
    {
        assert(inited);

        char tmp[512];
        sprintf_s(tmp, "Setup:\n"
            "\tCUDA: Blocks:%u Threads:%u Iteraions:%u\n"
            "\tEncrypted data size:%zd\n"
            "\tResults per batch:%zd\n"
            "\tDecryptors per batch:%zd\n"
            "\tConfig: %s",
            CudaBlocks(), CudaThreads(), CudaThreadIterations(),
            encrypted_data.size(), ResultsPerBatch(), KeysCheckedInBatch(), Config().toString().c_str());

        return std::string(tmp);
    }

private:

    void alloc()
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

    void free()
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

private:

    bool inited = false;

    uint32_t num_decryptors_per_batch = 0;

    //
    KernelInput kernel_inputs;

    // Constant per run
    std::vector<EncData> encrypted_data;

    // could be pretty much data here
    std::vector<Decryptor> decryptors;

    // could be pretty much data here
    std::vector<SingleResult> block_results;

    uint32_t CUDASetup [3] = {0};
};

