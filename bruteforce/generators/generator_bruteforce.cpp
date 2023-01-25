#include "generator_bruteforce.h"

#include "algorithm/keeloq/keeloq_kernel_input.h"
#include "kernels/kernel_result.h"


int GeneratorBruteforce::PrepareDecryptors(KeeloqKernelInput& inputs, uint16_t blocks, uint16_t threads)
{
	KernelResult generator_results;

	switch (inputs.generator.type)
	{
	case BruteforceType::Simple:
	{
		GeneratorBruteforceSimple::LaunchKernel(blocks, threads, inputs.ptr(), generator_results.ptr());
		break;
	}
	case BruteforceType::Filtered:
	{
		inputs.generator.next = inputs.generator.start;
		GeneratorBruteforceFiltered::LaunchKernel(blocks, threads, inputs.ptr(), generator_results.ptr());
		break;
	}
	case BruteforceType::Alphabet:
	{
		GeneratorBruteforceAlphabet::LaunchKernel(blocks, threads, inputs.ptr(), generator_results.ptr());
		break;
	}
	case BruteforceType::Pattern:
	{
		assert(false && "Not implemented");
		return 1;
		break;
	}
	case BruteforceType::Dictionary:
	default:
		return 0;
	}

	inputs.read();          // it will not cause underneath arrays copy
	generator_results.read();


	// last generated decryptor - is first on next batch
	//  Warning: In case of non-aligned calculations "real" last decryptor may be somewhere in the middle of array
	inputs.generator.next = inputs.decryptors->host_last().man;

	return generator_results.error;
}
