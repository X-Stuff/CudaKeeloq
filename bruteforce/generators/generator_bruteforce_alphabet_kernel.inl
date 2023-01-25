#include "device/cuda_context.h"

#include "kernels/kernel_result.h"
#include "algorithm/keeloq/keeloq_kernel_input.h"

#include "bruteforce/bruteforce_alphabet.h"
#include "bruteforce/bruteforce_type.h"


__global__ void DEFINE_GENERATOR_KERNEL(GeneratorBruteforceAlphabet, KeeloqKernelInput::TCudaPtr input, KernelResult::TCudaPtr resuls)
{
	CudaContext ctx = CudaContext::Get();

	assert(input->generator.type == BruteforceType::Alphabet);

	BruteforceAlphabet& alphabet = input->generator.alphabet;
	assert(alphabet.size() > 0);

	Decryptor& start = input->generator.start;

	CudaArray<Decryptor>& decryptors = *input->decryptors;

	// Imagine alphabet as rotating rings with letters on it
	// we have 8-bytes key so there will be 8 rings
	// indexes are per-byte and shows how much ring is rotated
	// and what 'letter' it should have.
	// Or also it can be considered as 8-digit N-based number
	uint64_t start_indexer = alphabet.lookup(start.man);

	// decomposed uint64 indexer for inner loop
	uint8_t curr_indexer[8];

	CUDA_FOR_THREAD_ID(ctx, decryptor_index, decryptors.num)
	{
		// Set current indexer to initial value first

		*(uint64_t*)curr_indexer = start_indexer;

		// Get new indexer value (rotate rings) depending on what decryptor we now producing (basically add 10-base number to N-base number)
		alphabet.add(curr_indexer, decryptor_index);

		Decryptor& current = decryptors[decryptor_index];
		uint8_t* pCurrentKey = (uint8_t*)&current.man;

		// produce key by indexes from curr_indexer
		UNROLL
		for (uint8_t i = 0; i < sizeof(uint64_t); ++i)
		{
			pCurrentKey[i] = alphabet[curr_indexer[i]];
		}
	}
}

DEFINE_GENERATOR_GETTER(GeneratorBruteforceAlphabet);