#include "device/cuda_context.h"

#include "kernels/kernel_result.h"
#include "algorithm/keeloq/keeloq_kernel_input.h"

#include "bruteforce/bruteforce_pattern.h"
#include "bruteforce/bruteforce_type.h"


__global__ void DEFINE_GENERATOR_KERNEL(GeneratorBruteforcePattern, KeeloqKernelInput::TCudaPtr input, KernelResult::TCudaPtr resuls)
{
    const BruteforceConfig& config = input->GetConfig();

	assert((config.type == BruteforceType::Alphabet) || (config.type == BruteforceType::Pattern));

	CudaContext ctx = CudaContext::Get();

	const BruteforcePattern& pattern = config.pattern;
	CudaArray<Decryptor>& decryptors = *input->decryptors;

	// Imagine alphabet as rotating rings with letters on it
	// we have 8-bytes key so there will be 8 rings
	// bytes in the key show how much ring is rotated
	// and what 'letter' it should have.
	// Or also it can be considered as 8-digit N-based number
	MultibaseNumber start = pattern.init(config.start.man());

    const uint32_t seed = config.start.seed();

	CUDA_FOR_THREAD_ID(ctx, decryptor_index, decryptors.num)
	{
        decryptors[decryptor_index] = Decryptor(pattern.next(start, decryptor_index).number(), seed);
	}
}

DEFINE_GENERATOR_GETTER(GeneratorBruteforcePattern);