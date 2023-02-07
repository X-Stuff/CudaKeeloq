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
