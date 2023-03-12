#pragma once

#include "common.h"

#include <cstring> // memcpy

#include "device/cuda_array.h"
#include "device/cuda_object.h"

#include "algorithm/keeloq/keeloq_encrypted.h"
#include "algorithm/keeloq/keeloq_decryptor.h"
#include "algorithm/keeloq/keeloq_single_result.h"
#include "algorithm/keeloq/keeloq_learning_types.h"

#include "bruteforce/bruteforce_config.h"


// Input data for main keeloq calculation kernel
struct KeeloqKernelInput : TGenericGpuObject<KeeloqKernelInput>
{
    // Constant per-run input data (captured encoded)
    CudaArray<EncParcel>* encdata = nullptr;

    // Single-run set of decryptors
    CudaArray<Decryptor>* decryptors = nullptr;

    // Single-run results
    CudaArray<SingleResult>* results = nullptr;

    KeeloqKernelInput() : TGenericGpuObject<KeeloqKernelInput>(this)
    {
    }

    KeeloqKernelInput(KeeloqKernelInput&& other) noexcept : TGenericGpuObject<KeeloqKernelInput>(this)
    {
        encdata = other.encdata;
        decryptors = other.decryptors;
        results = other.results;
        config = other.config;
        learnings = other.learnings;
    }

    KeeloqKernelInput& operator=(KeeloqKernelInput&& other) = delete;
    KeeloqKernelInput& operator=(const KeeloqKernelInput& other) = delete;

public:
    //
    __device__ __inline__ const KeeloqLearningType::Mask& GetLearningMask() const { return learnings; }

    //
    __device__ __inline__ bool AllLearningsEnabled() const { return allLearnings; }

    //
    __device__ __inline__ const BruteforceConfig& GetConfig() const { return config; }

public:
    void WriteDecryptors(const std::vector<Decryptor>& source, size_t from, size_t num);

    void NextDecryptor();

    void Initialize(const BruteforceConfig& inConfig, const KeeloqLearningType::Mask& inLearnings);

    // A "callback" which is called by generator. Used to prepare inputs for generators
    void BeforeGenerateDecryptors();

    // A "callback" which is called after generator creates Decryptors. Used to set correct last generated Decryptor
    void AfterGeneratedDecryptors();

    // Get Number of OTA inputs. Will do a GPU->CPU copy
    size_t NumInputs() const;

    // Does the fixed parts of inputs match? Will do a GPU->CPU copy
    bool InputsFixMatch() const;
private:
    // Which type of learning use for decryption
    KeeloqLearningType::Mask learnings;

    // optimizations. Just a bool field that could be accessed from GPU
    bool allLearnings = false;

    // from this decryptor generation will start
    BruteforceConfig config;
};
