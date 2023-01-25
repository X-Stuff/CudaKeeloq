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
	constexpr auto NumBlocks = 16;
	constexpr auto NumThreads = 32;

	auto testConfig = BruteforceConfig::GetAlphabet(0x6262626262626262, "abcd"_b, 0xFFFFFFFF);

	std::vector<Decryptor> decryptors(NumBlocks * NumThreads);
	KeeloqKernelInput generatorInputs(nullptr, CudaArray<Decryptor>::allocate(decryptors), nullptr, testConfig);
	KernelResult result;

	GeneratorBruteforceAlphabet::LaunchKernel(NumBlocks, NumThreads, generatorInputs.ptr(), result.ptr());

	generatorInputs.read();
	generatorInputs.decryptors->copy(decryptors);

	result.read();

	return true;
}