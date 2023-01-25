#pragma once

#include "common.h"

#include "device/generators/generator_bruteforce.h"

struct GeneratorBruteforceAlphabet : public IGenerator<GeneratorBruteforceAlphabet>
{
	static KernelFunc GetKernelFunctionPtr();
};