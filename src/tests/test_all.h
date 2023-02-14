#include "common.h"

#include <cuda_runtime_api.h>

#include "tests/test_alphabet.h"
#include "tests/test_benchmark.h"
#include "tests/test_console.h"
#include "tests/test_filters.h"
#include "tests/test_keeloq.h"
#include "tests/test_pattern.h"

namespace tests
{
	__host__ bool cuda_check_working();
}