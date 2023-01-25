#include "tests/test_alphabet.h"

#include "host/types/bruteforce_config.h"
#include "host/types/keeloq_decryptor.h"

#include "device/cuda_array.h"
#include "device/kernel_input.h"
#include "device/kernel_result.h"

#include "device/generators/generator_bruteforce_alphabet.h"

USE_NS_LOCATION

bool Tests::AlphabetGeneration()
{
	// Filtered generator test itself
	constexpr auto NumBlocks = 16;
	constexpr auto NumThreads = 32;

	auto testConfig = BruteforceConfig::GetAlphabet(0x6262626262626262, "abcd"_b, 0xFFFFFFFF);

	std::vector<Decryptor> decryptors(NumBlocks * NumThreads);
	KernelInput generatorInputs(nullptr, CudaArray<Decryptor>::allocate(decryptors), nullptr, testConfig);
	KernelResult result;

	GeneratorBruteforceAlphabet::LaunchKernel(NumBlocks, NumThreads, generatorInputs.ptr(), result.ptr());

	generatorInputs.read();
	generatorInputs.decryptors->copy(decryptors);

	result.read();

	return true;
}