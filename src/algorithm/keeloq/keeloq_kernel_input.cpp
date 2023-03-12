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
