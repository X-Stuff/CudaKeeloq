#pragma once

#include <cuda_runtime_api.h>

#include "common.h"

#include "tests/test_alphabet.h"
#include "tests/test_benchmark.h"
#include "tests/test_console.h"
#include "tests/test_filters.h"
#include "tests/test_generators.h"
#include "tests/test_keeloq.h"

namespace tests
{
    /** CPU ↔ GPU parity check for misc:: byte/bit helpers. Called at startup. */
    __host__ bool check_utils();

    /** Runs a trivial kernel to prove the GPU toolchain/driver works. */
    __host__ bool cuda_check_working();
}
