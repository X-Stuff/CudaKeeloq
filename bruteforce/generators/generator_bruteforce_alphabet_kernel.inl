#include "device/cuda_context.h"

#include "kernels/kernel_result.h"
#include "algorithm/keeloq/keeloq_kernel_input.h"

#include "bruteforce/bruteforce_alphabet.h"
#include "bruteforce/bruteforce_type.h"


__global__ void DEFINE_GENERATOR_KERNEL(GeneratorBruteforceAlphabet, KeeloqKernelInput::TCudaPtr input, KernelResult::TCudaPtr resuls)
{
	assert(input->generator.type == BruteforceType::Alphabet);

	CudaContext ctx = CudaContext::Get();

	const BruteforceAlphabet& alphabet = input->generator.alphabet;
	assert(alphabet.valid());

	const Decryptor& begin = input->generator.start;

	CudaArray<Decryptor>& decryptors = *input->decryptors;

	// Imagine alphabet as rotating rings with letters on it
	// we have 8-bytes key so there will be 8 rings
	// indexes are per-byte and shows how much ring is rotated
	// and what 'letter' it should have.
	// Or also it can be considered as 8-digit N-based number
	MultibaseNumber start = alphabet.cast(begin.man);


	CUDA_FOR_THREAD_ID(ctx, decryptor_index, decryptors.num)
	{
		decryptors[decryptor_index].man = alphabet.add(start, decryptor_index).number();
	}
}

DEFINE_GENERATOR_GETTER(GeneratorBruteforceAlphabet);