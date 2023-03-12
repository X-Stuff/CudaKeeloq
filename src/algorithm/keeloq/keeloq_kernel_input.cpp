#include "keeloq_kernel_input.h"
#include "common.h"

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
	config.next_decryptor();
}

void KeeloqKernelInput::Initialize(const BruteforceConfig& InConfig, const KeeloqLearningType::Mask& InLearnings)
{
    config = InConfig;
    learnings = InLearnings;
    allLearnings = learnings.is_all_enabled();
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
    config.last = decryptors->host_last();
}

size_t KeeloqKernelInput::NumInputs() const
{
    assert(encdata != nullptr && "Encdata unknown yet!");

    auto num = encdata ? encdata->host().num : 0;

    assert(num >= 1 && num <= 3 && "NumInputs(): Most probably something was wrong with memory copying!");

    return num;
}

bool KeeloqKernelInput::InputsFixMatch() const
{
    assert(encdata != nullptr && "Encdata unknown yet!");

    if (encdata)
    {
        std::vector<EncParcel> enc_data;
        encdata->copy(enc_data);

        assert(enc_data.size() >= 1 && enc_data.size() <= 3 && "InputsFixMatch(): Most probably something was wrong with memory copying!");

        if (enc_data.size() > 2)
        {
            return enc_data[0].fix() == enc_data[1].fix() && enc_data[1].fix() == enc_data[2].fix();
        }
        else if (enc_data.size() > 1)
        {
            return enc_data[0].fix() == enc_data[1].fix();
        }
    }

    return false;
}
