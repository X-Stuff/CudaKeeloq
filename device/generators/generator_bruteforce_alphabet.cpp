#include "generator_bruteforce_alphabet.h"

#include "generator_bruteforce_alphabet_kernel.inl"

GeneratorBruteforceAlphabet::KernelFunc GeneratorBruteforceAlphabet::GetKernelFunctionPtr()
{
	return GET_GENERATOR_KERNEL(GeneratorBruteforceAlphabet, KernelFunc);
}
