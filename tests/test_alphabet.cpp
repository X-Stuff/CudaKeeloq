#include "tests/test_alphabet.h"

#include "device/cuda_array.h"
#include "kernels/kernel_result.h"
#include "bruteforce/bruteforce_config.h"
#include "bruteforce/generators/generator_bruteforce.h"
#include "algorithm/keeloq/keeloq_kernel_input.h"
#include "algorithm/keeloq/keeloq_decryptor.h"


bool Tests::AlphabetGeneration()
{
	// Filtered generator test itself
	constexpr auto NumBlocks = 64;
	constexpr auto NumThreads = 64;

	auto testConfig = BruteforceConfig::GetAlphabet(0x0, "abcd"_b, 0xFFFFFFFF);

	std::vector<Decryptor> decryptors(NumBlocks * NumThreads);

	KeeloqKernelInput generatorInputs(nullptr, CudaArray<Decryptor>::allocate(decryptors), nullptr, testConfig);

	for (int i = 0; i < 16; ++i)
	{
		GeneratorBruteforce::PrepareDecryptors(generatorInputs, NumBlocks, NumThreads);

		generatorInputs.decryptors->copy(decryptors);

		assert((decryptors[0].man & 0x0000FFFFFFFFFFFF) == 0x616161616161);

		generatorInputs.NextDecryptor();
	}

	assert(decryptors[4095].man == 0x6464646464646464);
	return true;
}