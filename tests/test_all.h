#include "common.h"

#include <cuda_runtime_api.h>

#include "tests/test_alphabet.h"
#include "tests/test_filters.h"

namespace Tests
{
	__host__ bool CheckCudaIsWorking();
}