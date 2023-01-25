#pragma once

#include "common.h"

#include "device/generators/generator_bruteforce.h"

struct GeneratorBruteforceFiltered : public IGenerator<GeneratorBruteforceFiltered>
{
	static KernelFunc GetKernelFunctionPtr();
};