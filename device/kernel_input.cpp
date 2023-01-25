#include "kernel_input.h"

#include <algorithm>


void KernelInput::WriteDecryptors(const std::vector<Decryptor>& source, size_t from, size_t num)
{
	if (decryptors != nullptr)
	{
		assert(generator.type == BruteforceType::Dictionary);

		size_t copy_num = std::max(0ull, std::min(num, (source.size() - from)));
		decryptors->write(&source[from], copy_num);
	}
}


void KernelInput::NextDecryptor()
{
	assert(generator.type != BruteforceType::Dictionary);
	generator.next_decryptor();
}
