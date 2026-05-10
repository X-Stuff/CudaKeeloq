#include "algorithm/keeloq/keeloq_kernel_input.h"

#include "common.h"


void KeeloqKernelInput::InitInputsCache(const std::vector<EncParcel>& inputs)
{
    // COPY TO CONSTANT CACHE ON GPU
    inputsCount = static_cast<uint8_t>(inputs.size());
    assert(inputsCount >= 1 && inputsCount <= 3 && "Invalid number of inputs. Must be between 1 and 3");

    if (inputsCount > 2)
    {
        inputsFixMatch = inputs[0].fix() == inputs[1].fix() && inputs[1].fix() == inputs[2].fix();
    }
    else if (inputsCount > 1)
    {
        inputsFixMatch = inputs[0].fix() == inputs[1].fix();
    }
    else
    {
        inputsFixMatch = true;
    }

    // Constant per run
    CudaFixedArray<EncParcel, 3> encrypted_array;
    for (uint8_t i = 0; i < encrypted_array.size(); i++)
    {
        encrypted_array[i] = i < inputs.size() ? inputs[i] : EncParcel{};
    }

    CudaFixedArray<EncParcel, 3>::constantCopy(InputsCache, encrypted_array);
}

size_t KeeloqKernelInput::BytesAllocated() const
{
    return (decryptors ? decryptors->allocated() : 0) + (results ? results->allocated() : 0);
}

void KeeloqKernelInput::WriteDecryptors(const std::vector<Decryptor>& source, size_t from, size_t num)
{
	if (decryptors != nullptr)
	{
		assert(config.type == BruteforceType::Dictionary);

		size_t copy_num = std::max<size_t>(0, std::min(num, (source.size() - from)));
		decryptors->write(&source[from], copy_num);
	}
}


void KeeloqKernelInput::NextDecryptor()
{
	assert(config.type != BruteforceType::Dictionary);
	config.nextDecryptor();
}

void KeeloqKernelInput::Initialize(const BruteforceConfig& inConfig, const std::vector<EncParcel>& inInputs)
{
    config = inConfig;
    InitInputsCache(inInputs);
}

void KeeloqKernelInput::BruteforcePrepare(const KeeloqLearning::Matrix& inLearnings, InputsMutation mutations)
{
    assert(is_valid(mutations) && "Invalid input mutation mask");
    assert(config.type != BruteforceType::XorFix || !!(mutations & InputsMutation::XorFix) &&
        "In XorFix bruteforce you should have always XorFix mutation enabled");

    learnings = inLearnings;
    allLearnings = learnings.isAllEnabled();
    mutationsMask = mutations;
    readyForBrute = true;
}

void KeeloqKernelInput::BeforeGenerateDecryptors()
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

void KeeloqKernelInput::AfterGeneratedDecryptors()
{
    // last generated decryptor - is first on next batch
    //  Warning: In case of non-aligned calculations "real" last decryptor may be somewhere in the middle of array
    config.last = decryptors->hostLast();
}

uint8_t KeeloqKernelInput::NumInputs() const
{
    return inputsCount;
}

void KeeloqKernelInput::read()
{
    TGenericGpuObject<KeeloqKernelInput>::read();
    readyForBrute = false;
}
