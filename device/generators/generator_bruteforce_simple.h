#pragma once

#include "common.h"

#include "device/generators/generator_bruteforce.h"

struct GeneratorBruteforceSimple : public IGenerator<GeneratorBruteforceSimple>
{
	static KernelFunc GetKernelFunctionPtr();
};