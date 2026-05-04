#pragma once

#include <cstring>

#include "common.h"

#include "device/cuda_array.h"
#include "device/cuda_fixed_array.h"
#include "device/cuda_object.h"

#include "algorithm/keeloq/keeloq_decryptor.h"
#include "algorithm/keeloq/keeloq_encrypted.h"
#include "algorithm/keeloq/keeloq_learning_types.h"
#include "algorithm/keeloq/keeloq_single_result.h"

#include "bruteforce/bruteforce_config.h"


// Constant per-run input data (captured encoded)
extern __constant__ CudaFixedArray<EncParcel, 3> InputsCache;

/**
 * Host/device-shared input bundle for the main keeloq bruteforce kernel.
 * Owns the per-run decryptor batch, result buffer, learning matrix, and bruteforce config.
 */
struct KeeloqKernelInput : TGenericGpuObject<KeeloqKernelInput>
{
    uint8_t inputsCount = 0;

    // Single-run set of decryptors
    CudaArray<Decryptor>* decryptors = nullptr;

    // Single-run results
    CudaArray<SingleResult>* results = nullptr;

    KeeloqKernelInput() : TGenericGpuObject<KeeloqKernelInput>(this)
    {
    }

    KeeloqKernelInput(KeeloqKernelInput&& other) noexcept : TGenericGpuObject<KeeloqKernelInput>(this)
    {
        inputsCount = other.inputsCount;
        decryptors = other.decryptors;
        results = other.results;
        config = other.config;
        learnings = other.learnings;
    }

    KeeloqKernelInput& operator=(KeeloqKernelInput&& other) = delete;
    KeeloqKernelInput& operator=(const KeeloqKernelInput& other) = delete;

public:
    /** Learning-type selection matrix for the current run. */
    __device__ __host__ __inline__ const KeeloqLearning::Matrix& GetLearningMatrix() const { return learnings; }

    /** Fast path indicator: true when every learning entry is enabled. */
    __device__ __host__ __inline__ bool AllLearningsEnabled() const { return allLearnings; }

    /** Active bruteforce configuration (device-side access). */
    __device__ __inline__ const BruteforceConfig& GetConfig() const { return config; }

    /** Uploads encrypted inputs to the device-side constant cache. */
    __device__ void InitInputsCache(const std::vector<EncParcel>& inputs);

    /** Bytes allocated by the decryptor and result buffers (for metrics). */
    size_t BytesAllocated() const;

public:
    /** Copies a slice from a host-side decryptor dictionary into the device buffer. */
    void WriteDecryptors(const std::vector<Decryptor>& source, size_t from, size_t num);

    /** Advances the bruteforce start key by one generator step (non-dictionary modes). */
    void NextDecryptor();

    /** One-time setup: captures config, learning matrix, and uploads inputs. */
    void Initialize(const BruteforceConfig& inConfig, const std::vector<EncParcel>& inInputs, const KeeloqLearning::Matrix& inLearnings);

    /** Generator callback: prepares host-side state just before a decryptor batch is generated. */
    void BeforeGenerateDecryptors();

    /** Generator callback: captures the last generated decryptor as the next batch's starting point. */
    void AfterGeneratedDecryptors();

    /** Returns the OTA input count (performs a GPU→CPU copy). */
    size_t NumInputs() const;

    /** True if every input in the batch shares the same fixed code. */
    bool InputsFixMatch() const { return inputsFixMatch; }

private:
    // Which type of learning use for decryption
    KeeloqLearning::Matrix learnings;

    // optimizations. Just a bool field that could be accessed from GPU
    bool allLearnings = false;

    // optimizations. Flag that shows that all fixed parts of inputs match.
    bool inputsFixMatch = false;

    // from this decryptor generation will start
    BruteforceConfig config;
};
