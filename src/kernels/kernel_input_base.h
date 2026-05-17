#pragma once

#include <cstring>
#include <functional>
#include <memory>

#include "common.h"

#include "device/cuda_array.h"
#include "device/cuda_object.h"
#include "device/cuda_fixed_array.h"

#include "algorithm/keeloq/keeloq_decryptor.h"
#include "algorithm/keeloq/keeloq_encrypted.h"

#include "bruteforce/bruteforce_config.h"
#include "bruteforce/bruteforce_result.h"


// Constant per-run input data (captured encoded)
extern __constant__ CudaFixedArray<EncParcel, 3> InputsCache;

struct BruteforceConfig;

/**
 *  The common interface for all KeeloqKernelInput types, providing shared state and generator callbacks.
 * Includes BruteforceConfig and Decryptors, ancestors differs only by results and own private data
 */
struct IKeeloqKernelInputBase
{
    /**
     *  Type of inputs.
     *      Multi - inputs when kernel calculates all learnings with modifications and single input transform at one call
     *      Single - inputs when kernel calculates only one learning and one transform and one modification at call
     */
    enum Type { Single, Multi };

    using GetMatchProgressCallback = std::function<void(size_t current, size_t total)>;

    using Ptr = IKeeloqKernelInputBase*;

    virtual ~IKeeloqKernelInputBase()
    {
        FreeGPU();
    }

public:
    /** Copies self object (shallow copy) to GPU */
    virtual __host__ IKeeloqKernelInputBase::Type type() const = 0;

    /** Get first matched result, final ancestors should return a match by searching own results array */
    virtual __host__ BruteforceResult getMatch(GetMatchProgressCallback onProgress = nullptr) const = 0;

    /** Get specific result by index */
    virtual __host__ BruteforceResult getResult(size_t index) const = 0;

    /** Prepare inputs for the next batch, basically set up internal fields so they become valid in kernels */
    virtual __host__ void prepareBatch(const KeeloqLearning::Matrix& learningMatrix, InputTransform inputMutations) = 0;

public:
    /** Copies self object (shallow copy) to GPU */
    virtual __host__ Ptr gpu() = 0;

    /** Copies self object (shallow copy) from GPU */
    virtual __host__ cudaError_t sync() = 0;

public:
    /**
     *  Allocates GPU memory for the decryptors (num in batch), and results (ancestors do)
     *
     *  @param totalNumDecryptors       Total number of decryptors to allocate
     *  @param numInputs                Number of inputs to calculate number of results
     */
    virtual __host__ cudaError_t AllocateGPU(size_t totalNumDecryptors, uint8_t numInputs);

    /** Frees GPU memory allocated for decryptors and results (ancestors do) */
    virtual __host__ void FreeGPU();

    /** Bytes allocated by the decryptors (for metrics), ancestors should add their allocations */
    virtual __host__ size_t BytesAllocated() const;

    /** One-time setup: captures config, uploads inputs. */
    virtual __host__ void Initialize(const BruteforceConfig& inConfig, const std::vector<EncParcel>& inInputs);

    /** Generator callback: prepares host-side state just before a decryptor batch is generated. */
    virtual __host__ void BeforeGenerateDecryptors();

    /** Generator callback: captures the last generated decryptor as the next batch's starting point. */
    virtual __host__ void AfterGeneratedDecryptors();

public:
    // Single-run set of decryptors
    __device__ __host__ __inline__ CudaArray<Decryptor>* Decryptors() { return decryptors; }

    // Number of inputs in the current run
    __device__ __host__ __inline__  uint8_t InputsCount() const { return inputsCount; }

    // Return bruteforce configuration for current run
    __device__ __host__ __inline__ const BruteforceConfig& GetConfig() const { return config; }

    /** True if every input in the batch shares the same fixed code. */
    __device__ __host__ __inline__ bool InputsFixMatch() const { return inputsFixMatch; }

    /** Copies a slice from a host-side decryptor dictionary into the device buffer. */
    __device__ __host__ void WriteDecryptors(const std::vector<Decryptor>& source, size_t from, size_t num);

    /** Advances the bruteforce start key by one generator step (non-dictionary modes). */
    __device__ __host__ void NextDecryptor();

public:
    static void InitInputsCache(const std::vector<EncParcel>& inputs);

    template<typename TDerived>
    static std::unique_ptr<IKeeloqKernelInputBase> Create() { return std::make_unique<TDerived>(); }

protected:
    /** A helper method for ancestor to get input for bruteforce match struct */
    __host__ const EncParcel& GetInput(size_t index) const { return inputs[index]; }

public:
    uint8_t inputsCount = 0;

    // Single-run set of decryptors
    CudaArray<Decryptor>* decryptors = nullptr;

private:
    // Config for current bruteforce run, captured on host and used on device
    BruteforceConfig config;

    // Used inputs, this is cached for CPU only, GPU uses its own global cache
    std::vector<EncParcel> inputs;

    // True if all inputs have the same fixed part (not matched fixed parts are not allowed)
    bool inputsFixMatch = false;
};

/**
 *
 */
template<typename TDerived>
struct TKeeloqKernelInputBase : public TGenericGpuObject<TDerived>, public IKeeloqKernelInputBase
{
public:
    using TGenericGpuObject<TDerived>::TGenericGpuObject;

    TKeeloqKernelInputBase(TKeeloqKernelInputBase&& other, TDerived* newSelf) noexcept :
        TGenericGpuObject<TDerived>(std::move(other), newSelf), IKeeloqKernelInputBase(other) // shallow copy base fields
    {
        readyForBrute = other.readyForBrute;
        other.readyForBrute = false;
    }

public:
    /** Copies self object (shallow copy) to GPU, returns as pointer to base interface */
    virtual __host__ IKeeloqKernelInputBase::Ptr gpu() final override { return TGenericGpuObject<TDerived>::ptr(); };

    /** Copies self object (shallow copy) from GPU */
    virtual __host__ cudaError_t sync() final override { return read(); };

public:
    /** True if ready for bruteforce (prepare called) */
    __host__ bool Ready() const { return readyForBrute; }

public:
    __host__ virtual cudaError_t read() override
    {
        const auto error = TGenericGpuObject<TDerived>::read();
        readyForBrute = false;

        return error;
    }

protected:
    /** call to set ready state right before bruteforce */
    __host__ void SetReady(bool value) { readyForBrute = value; }

private:
    // Flag resets every time kernel finished, and sets on Prepare
    // Safety flag
    bool readyForBrute = false;
};

std::string_view toString(IKeeloqKernelInputBase::Type type);