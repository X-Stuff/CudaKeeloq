#pragma once

#include <vector>
#include <assert.h>

#include "keeloq_main.cuh"



constexpr char WAIT_SPIN[] = "|/-\\";

#define WAIT_CHAR(i) (WAIT_SPIN[i % (sizeof(WAIT_SPIN) - 1)])

struct CudaRunSetup
{
    CudaRunSetup(const std::vector<EncData>& data, const DectyptorGenerationConfig& gen, uint32_t blocks, uint32_t threads, uint32_t iterations)
        : encrypted_data(data)
    {
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

    inline const DectyptorGenerationConfig& Config() const { assert(inited); return kernel_inputs.generator; }

    inline GeneratorType Type() { assert(inited); return Config().type; }

    inline KernelInput& Inputs() { assert(inited); return kernel_inputs; }

    inline size_t NumBatches() {
        assert(inited);
        if (Type() == GeneratorType::Dictionary) {
            return Config().dict_size() / KeysCheckedInBatch() + 1;
        }
        else {
            return Config().brute_size() / KeysCheckedInBatch() + 1;
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
        sprintf_s(tmp, "Setup:\n\tEncrypted data size:%zd\n\tResults per batch:%zd\n\tDecryptors per batch:%zd\n\tConfig: %s",
            encrypted_data.size(), ResultsPerBatch(), KeysCheckedInBatch(), Config().ToString().c_str());

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
};

