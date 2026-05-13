#include "kernels/kernel_input_base.h"

#include "bruteforce/bruteforce_type.h"
#include "bruteforce/bruteforce_config.h"


void IKeeloqKernelInputBase::InitInputsCache(const std::vector<EncParcel>& inputs)
{
    // Constant per run
    CudaFixedArray<EncParcel, 3> encrypted_array;
    for (uint8_t i = 0; i < encrypted_array.size(); i++)
    {
        encrypted_array[i] = i < inputs.size() ? inputs[i] : EncParcel{};
    }

    CudaFixedArray<EncParcel, 3>::constantCopy(InputsCache, encrypted_array);
}


cudaError_t IKeeloqKernelInputBase::AllocateGPU(size_t totalNumDecryptors, uint8_t numInputs)
{
    assert(numInputs > 0 && numInputs == inputsCount &&
        "Invalid number provided to alloc function, should match number what used in initialization");

    assert(decryptors == nullptr && "Decryptors data already allocated on GPU");

    if (decryptors == nullptr)
    {
        decryptors = CudaArray<Decryptor>::allocate(totalNumDecryptors);
    }

    return decryptors != nullptr ? cudaSuccess : cudaGetLastError();
}


void IKeeloqKernelInputBase::FreeGPU()
{
    if (decryptors != nullptr)
    {
        decryptors->free();
        decryptors = nullptr;
    }
}

size_t IKeeloqKernelInputBase::BytesAllocated() const
{
    return (decryptors ? decryptors->allocated() : 0);
}

void IKeeloqKernelInputBase::Initialize(const BruteforceConfig& inConfig, const std::vector<EncParcel>& inInputs)
{
    config = inConfig;
    inputs = inInputs;
    InitInputsCache(inInputs);

    inputsCount = static_cast<uint8_t>(inInputs.size());
    assert(inputsCount >= 1 && inputsCount <= 3 && "Invalid number of inputs. Must be between 1 and 3");

    if (inputsCount > 2)
    {
        inputsFixMatch = inInputs[0].fix() == inInputs[1].fix() && inInputs[1].fix() == inInputs[2].fix();
    }
    else if (inputsCount > 1)
    {
        inputsFixMatch = inInputs[0].fix() == inInputs[1].fix();
    }
    else
    {
        inputsFixMatch = true;
    }
}

void IKeeloqKernelInputBase::BeforeGenerateDecryptors()
{
    switch (config.type)
    {
    case BruteforceType::Filtered:
    {
        config.filters.sync_key = config.start.man();
        break;
    }
    default:
        break;
    }
}

void IKeeloqKernelInputBase::AfterGeneratedDecryptors()
{
    // last generated decryptor - is first on next batch
    //  Warning: In case of non-aligned calculations "real" last decryptor may be somewhere in the middle of array
    config.last = Decryptors()->hostLast();
}

void IKeeloqKernelInputBase::WriteDecryptors(const std::vector<Decryptor>& source, size_t from, size_t num)
{
    if (decryptors != nullptr)
    {
        assert(config.type == BruteforceType::Dictionary);

        size_t copy_num = std::max<size_t>(0, std::min(num, (source.size() - from)));
        decryptors->write(&source[from], copy_num);
    }
}

void IKeeloqKernelInputBase::NextDecryptor()
{
    assert(config.type != BruteforceType::Dictionary);
    config.nextDecryptor();
}
